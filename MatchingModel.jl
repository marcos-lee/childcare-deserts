module MatchingModel

using Distributions

function gen_data(N_g, N_p, break_point)
    dists = rand(truncated(LogNormal(2.75, 0.6), 2, 50), N_g, N_p)
    subsidy = rand(Uniform(0,1), N_p) .< .8
    qual = (rand(Uniform(0,1), N_p) .< 0.2)
    home = rand(Uniform(0,1), N_p) .< 0.4

    family_pov = [dists repeat(subsidy', N_g) repeat(qual', N_g) repeat(home', N_g)]

    vacancy_pov = [dists' repeat(subsidy, 1, N_g)]

    choice_set = dists .< break_point

    return family_pov, vacancy_pov, choice_set
end

function matching_function(A, NS_f, NS_v)
    X = A * (NS_f^0.5) * (NS_v^0.5)
    return min(X, NS_f, NS_v)
end

function μ_f(subsidy, qual, home, dist, α, β_f)
    util = α[1] + α[2] * subsidy + α[3] * qual + α[4] * home - cost(dist, β_f) /100
end

function lnEU_f(subsidy, qual, home, dist, p_match, α, β_f)
    if p_match == 0
        lnEU_f = -99999.0
    else
        lnEU_f = μ_f(subsidy, qual, home, dist, α, β_f) + log(p_match)
    end
    return lnEU_f
end

function cost(dist, β)
    return dist ^ β
end

function lnEU_v(subsidy, dist, p_match, β, R)
    if p_match == 0
        lnEU_v = -99999.0
    else
        lnEU_v = log(p_match) + R * subsidy - cost(dist, β) / 100
    end
    return lnEU_v
end

function ϕ_v(subsidy_subset, dist_subset, p_match, β_v, R)
    temp = lnEU_v.(subsidy_subset, dist_subset, p_match, β_v, R)
    if sum(exp.(temp)) == 0
        print("divide by zero!")
    end
    ϕ_v = exp.(temp) ./ sum(exp.(temp))
    return ϕ_v
end

function ϕ(lnEU, σ, choice_set)
    if sum(exp.(lnEU)) == 0.0
        print("divide by zero!")
    end
    ϕ = exp.((lnEU./ σ)) ./ sum(exp.((lnEU./ σ) ).* choice_set )
    return ϕ
end

function ϕ_f(subsidy_subset, qual_subset, home_subset, dist_subset, p_match, α, β_f)
    temp = lnEU_f.(subsidy_subset, qual_subset, home_subset, dist_subset, p_match, Ref(α), β_f)
    if sum(exp.(temp)) == 0.0
        print("divide by zero!")
    end
    ϕ_f = exp.(temp) ./ sum(exp.(temp))
    return ϕ_f
end

function unpack(param)
    α = param[1:4]
    β_v = param[5]
    β_f = param[6]
    R = param[7]
    A = param[8]
    σ = param[9]
    return α, β_f, β_v, R, A, σ
end


function gen_shares(vacancy_pov, family_pov, p_match_f, p_match_v, param, N_g, N_p, choice_set)
    α, β_v, β_f, R, A, σ = unpack(param)
    share_f = Array{Float64,2}(undef, N_g, N_p) #will contain 10 arrays of 500x1
    share_v = Array{Float64,2}(undef, N_p, N_g) #will contain 500 arrays of 10x1
    subsidy_f = family_pov[:, (N_p+1):2*N_p]
    qual = family_pov[:, (2*N_p+1):3*N_p]
    home = family_pov[:, (3*N_p+1):4*N_p]
    dist_f = family_pov[:, 1:N_p]
    subsidy_v = vacancy_pov[:, (N_g+1):2*N_g]
    dist_v = vacancy_pov[:, (1):N_g]
    for i = 1:N_g
        #temp = family_pov[family_pov.zipcode .== i,:]
        lnEU = lnEU_f.(subsidy_f[i,:], qual[i,:], home[i,:], dist_f[i,:], p_match_f[i,:], Ref(α), β_f)
        share_f[i,:] = ϕ(lnEU, σ, choice_set[i,:])
    end
    for i = 1:N_p
        #temp = vacancy_pov[vacancy_pov.id .== i,:]
        lnEU = lnEU_v.(subsidy_v[i,:], dist_v[i,:], p_match_v[i,:], β_v, R)
        share_v[i,:] = ϕ(lnEU, σ, choice_set[:,i])
    end
    return share_f, share_v
end

function get_equilibrium(vacancy_pov, family_pov, param, N_f, N_v, choice_set)
    N_p = size(vacancy_pov, 1)
    N_g = size(family_pov, 1)
    share_f = zeros(N_g, N_p)
    share_v = zeros(N_p, N_g)
    #share_f = Array{Array{Float64,1},1}(undef, N_g) #will contain 10 arrays of 500x1
    #share_v = Array{Array{Float64,1},1}(undef, N_p) #will contain 500 arrays of 10x1
    #N_f = 1000 #Array{Float64,1}(undef, 10) #will contain 10 arrays of 500x1
    #N_v = 50 #Array{Float64,1}(undef, 500) #will contain 500 arrays of 10x1
    for i = 1:N_g
        share_f[i,:] .= (1/sum(choice_set[i,:])) .* choice_set[i,:]
        #N_f[i] = 1_000
    end
    for i = 1:N_p
        share_v[i,:] .= (1/sum(choice_set[:,i])) .* choice_set[:,i]
        #N_v[i] = 20
    end
    counter = 0
    α, β_v, β_f, R, A, σ = unpack(param)
    #share_f, share_v = gen_shares(vacancy_pov, family_pov, p_match_f, p_match_v, param)
    #fx, p_match_f, p_match_v, share_f_new, share_v_new = fx_once(share_f, share_v, p_match_f, p_match_v, N_f, N_v, A, γ)
    fx = 1.0
    #share_f_new = 1.0
    #share_v_new = 1.0
    while fx > 0.0001
        if counter >= 200
             break
         end
        p_match_f, p_match_v = fx_once(share_f, share_v, N_f, N_v, N_g, N_p, A, choice_set)
        share_f_new, share_v_new = gen_shares(vacancy_pov, family_pov, p_match_f, p_match_v, param, N_g, N_p, choice_set)
        print("\t", "Iteration ", counter, "\n")
        fx_f = maximum(abs.(share_f[choice_set] .- share_f_new[choice_set]))
        fx_v = maximum(abs.(share_v[choice_set'] .- share_v_new[choice_set']))
        #for i = 1:10
        #    fx_f[i] = maximum(abs.(share_f[i] .- share_f_new[i]))
        #end
        #for i = 1:500
        #    fx_v[i] = maximum(abs.(share_v[i] .- share_v_new[i]))
        #end
        print("\t")
        print(maximum(fx_f), "\t")
        print(maximum(fx_v), "\t")
        #share_f_fx = maximum(abs.(reduce(vcat, share_f .- share_f_new)))
        #share_v_fx = maximum(abs.(reduce(vcat, share_v .- share_v_new)))
        fx = max(fx_f, fx_v)
        share_f = copy(share_f_new)
        share_v = copy(share_v_new)
        print(fx, "\n")
        counter += 1
    end
    return share_f, share_v
end

function fx_once(share_f, share_v, N_f, N_v, N_g, N_p, A, choice_set)

    NS_f = share_f .* N_f
    NS_v = share_v .* N_v

    #NS_v = deepcopy(share_v)
    #NS_f = deepcopy(share_f)

    p_match_f = Array{Float64,2}(undef, N_g, N_p)
    p_match_v = Array{Float64,2}(undef, N_p, N_g)
    for f = 1:N_g
        for v = 1:N_p
            #NS_f[f][v] = share_f[f][v] * N_f[f]
            #NS_v[v][f] = share_v[v][f] * N_v[v]
            if choice_set[f,v] == 0
                p_match_f[f,v] = 0
                p_match_v[v,f] = 0
            else
                if NS_f[f, v] == 0
                    print("NSf = 0")
                    p_match_f[f, v] = 0
                else
                    p_match_f[f, v] = matching_function(A, NS_f[f, v], NS_v[v, f]) / NS_f[f, v]
                end
                if NS_v[v, f] == 0
                    print("NSv = 0", "\t", v, "\t", f)
                    p_match_v[v, f] = 0
                else
                    p_match_v[v, f] = matching_function(A, NS_f[f, v], NS_v[v, f]) / NS_v[v, f]
                end
            end
            #matches[i] = MatchingModel.matching_function(A, γ, NS_f[f][v], NS_v[v][f])
        end
    end
    #share_f_new, share_v_new = gen_shares(vacancy_pov, family_pov, p_match_f, p_match_v, param)

    #share_f_fx = maximum(abs.(reduce(vcat, share_f) .- reduce(vcat, share_f_new)))
    #share_v_fx = maximum(abs.(reduce(vcat, share_v) .- reduce(vcat, share_v_new)))
    #fx = max(share_f_fx, share_v_fx)
    return p_match_f, p_match_v
end

end
