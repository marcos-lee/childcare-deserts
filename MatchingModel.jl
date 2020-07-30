module MatchingModel

function matching_function(A, NS_f, NS_v)
    X = (A) * (NS_f^0.5) * (NS_v^0.5)
    return min(X, NS_f, NS_v)
end

function μ_f(subsidy, qual, home, dist, α, β_f)
    util = α[1] + α[2] * subsidy + α[3] * qual + α[4] * home - cost(dist, β_f) /100
end

function lnEU(subsidy, home, p_match, α)
    util = α[1] + α[2] * subsidy + α[3] * home + log(p_match)
end

function ϕ(lnEU, σ)
    ϕ = exp.(lnEU ./ σ) ./ sum(exp.(lnEU ./ σ))
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

function ϕ_f(subsidy_subset, qual_subset, home_subset, dist_subset, p_match, α, β_f)
    temp = lnEU_f.(subsidy_subset, qual_subset, home_subset, dist_subset, p_match, Ref(α), β_f)
    if sum(exp.(temp)) == 0.0
        print("divide by zero!")
    end
    ϕ_f = exp.(temp) ./ sum(exp.(temp))
    return ϕ_f
end

function unpack(param)
    αf = param[1:3]
    αv = param[4:6]
    A = param[end]
    return αf, αv, A
end
function gen_shares(vacancy_pov, family_pov, p_match_f, p_match_v, param, ntypes)
    αf, αv, A = unpack(param)
    share_f = Array{Float64,2}(undef, ntypes, ntypes) #will contain 10 arrays of 500x1
    share_v = Array{Float64,2}(undef, ntypes, ntypes) #will contain 500 arrays of 10x1
    for i = 1:ntypes
        lneu = lnEU.(family_pov[i, 1:ntypes], family_pov[i, (ntypes+1):(2*ntypes)], p_match_f[i,:], Ref(αf))
        share_f[i,:] = MatchingModel.ϕ(lneu, 2.0)
    end
    for i = 1:ntypes
        lneu = lnEU.(vacancy_pov[i, 1:ntypes], vacancy_pov[i, (ntypes+1):(2*ntypes)], p_match_v[i,:], Ref(αv))
        share_v[i,:] = MatchingModel.ϕ(lneu, 2.0)
    end
    return share_f, share_v
end
function gen_shares(vacancy_pov, family_pov, p_match_f, p_match_v, param, N_g, N_p)
    α, β_v, β_f, R, A, γ = unpack(param)
    share_f = Array{Float64,2}(undef, N_g, N_p) #will contain 10 arrays of 500x1
    share_v = Array{Float64,2}(undef, N_p, N_g) #will contain 500 arrays of 10x1
    for i = 1:N_g
        temp = family_pov[family_pov.zipcode .== i,:]
        share_f[i,:] = ϕ_f(temp.subsidy, temp.qual, temp.home, temp.dist, p_match_f[i], α, β_f)
    end
    for i = 1:N_p
        temp = vacancy_pov[vacancy_pov.id .== i,:]
        share_v[i,:] = ϕ_v(temp.subsidy, temp.dist, p_match_v[i], β_v, R)
    end
    return share_f, share_v
end
function get_equilibrium(vacancy_pov, family_pov, param, ntypes)
    #share_f = Array{Float64,2}(undef, 10, 500) #will contain 10 arrays of 500x1
    #share_v = Array{Float64,2}(undef, 500, 10) #will contain 500 arrays of 10x1
    Ns = 2500
    share_f = (1/ntypes) * ones(ntypes, ntypes)
    share_v = (1/ntypes) * ones(ntypes, ntypes)
    #N_f = 1000 #Array{Float64,1}(undef, 10) #will contain 10 arrays of 500x1
    #N_v = 50 #Array{Float64,1}(undef, 500) #will contain 500 arrays of 10x1
    #for i = 1:10
    #    share_f[i, :] = (1/500)*ones(500)
    #    #N_f[i] = 1_000
    #end
    #for i = 1:500
    #    share_v[i, :] = (1/10)*ones(10)
    #    #N_v[i] = 20
    #end
    counter = 0
    αf, αv, A = unpack(param)
    #share_f, share_v = gen_shares(vacancy_pov, family_pov, p_match_f, p_match_v, param)
    #fx, p_match_f, p_match_v, share_f_new, share_v_new = fx_once(share_f, share_v, p_match_f, p_match_v, N_f, N_v, A, γ)
    fx = 1.0
    #share_f_new = 1.0
    #share_v_new = 1.0
    while fx > 0.0001
        p_match_f, p_match_v = fx_once(share_f, share_v, Ns, A, ntypes)
        share_f_new, share_v_new = gen_shares(vacancy_pov, family_pov, p_match_f, p_match_v, param, ntypes)
        print("\t", "Iteration ", counter, "\n")
        fx_f = maximum(abs.(share_f .- share_f_new))
        fx_v = maximum(abs.(share_v .- share_v_new))
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
function get_equilibrium(vacancy_pov, family_pov, param, N_f, N_v)
    #share_f = Array{Float64,2}(undef, 10, 500) #will contain 10 arrays of 500x1
    #share_v = Array{Float64,2}(undef, 500, 10) #will contain 500 arrays of 10x1
    N_p = convert(Int, maximum(vacancy_pov.id))
    N_g = convert(Int, maximum(vacancy_pov.zipcode))
    share_f = (1/N_p) * ones(N_g, N_p)
    share_v = (1/N_g) * ones(N_p, N_g)
    #N_f = 1000 #Array{Float64,1}(undef, 10) #will contain 10 arrays of 500x1
    #N_v = 50 #Array{Float64,1}(undef, 500) #will contain 500 arrays of 10x1
    #for i = 1:10
    #    share_f[i, :] = (1/500)*ones(500)
    #    #N_f[i] = 1_000
    #end
    #for i = 1:500
    #    share_v[i, :] = (1/10)*ones(10)
    #    #N_v[i] = 20
    #end
    counter = 0
    α, β_v, β_f, R, A, γ = unpack(param)
    #share_f, share_v = gen_shares(vacancy_pov, family_pov, p_match_f, p_match_v, param)
    #fx, p_match_f, p_match_v, share_f_new, share_v_new = fx_once(share_f, share_v, p_match_f, p_match_v, N_f, N_v, A, γ)
    fx = 1.0
    #share_f_new = 1.0
    #share_v_new = 1.0
    while fx > 0.0001
        p_match_f, p_match_v = fx_once(share_f, share_v, N_f, N_v, N_g, N_p, A, γ)
        share_f_new, share_v_new = gen_shares(vacancy_pov, family_pov, p_match_f, p_match_v, param, N_g, N_p)
        print("\t", "Iteration ", counter, "\n")
        fx_f = maximum(abs.(share_f .- share_f_new))
        fx_v = maximum(abs.(share_v .- share_v_new))
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

function fx_once(share_f, share_v, Ns, A, ntypes)

    NS_f = share_f .* Ns
    NS_v = share_v .* Ns

    #NS_v = deepcopy(share_v)
    #NS_f = deepcopy(share_f)
    p_match_f = Array{Float64,2}(undef, ntypes, ntypes)
    p_match_v = Array{Float64,2}(undef, ntypes, ntypes)
    for f = 1:ntypes
        for v = 1:ntypes
            #NS_f[f][v] = share_f[f][v] * N_f[f]
            #NS_v[v][f] = share_v[v][f] * N_v[v]
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
