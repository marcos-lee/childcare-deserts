module GenSample

include("MatchingModel.jl")
using .MatchingModel
using Distributions

function y_search(share, index_search, N_sub)
    N1 = size(share, 1)
    cumshare = cumsum(share, dims = 2)
    res = Vector{Vector{Int64}}(undef, N1)
    for i = 1:N1
        temp = Vector{Int64}(undef, N_sub[i])
        for k = 1:N_sub[i]
            temp[k] = who_searches(index_search[i][k], cumshare[i,:])
        end
        res[i] = temp
    end
    return res
end
function y_match(p_match, index_match, N_sub, y_search)
    N1 = size(p_match, 1)
    res = Vector{Vector{Int64}}(undef, N1)
    for i = 1:N1
        temp = Vector{Int64}(undef, N_sub[i])
        for k = 1:N_sub[i]
            temp[k] = y_search[i][k] * (index_match[i][k] < p_match[i,y_search[i][k]])
        end
        res[i] = temp
    end
    return res
end
function who_searches(index_search, cumshare)
    out = Vector{Bool}(undef, size(cumshare, 1))
    out[1] = (index_search < cumshare[1])
    for i = 1:size(out,1)-1
        out[i+1] = iswithin(index_search, cumshare[i:(i+1)])
    end
    #out[end] = (index_search >= cumshare[end])
    pos = first(findall(x -> x == 1, out))
    return pos
end
function iswithin(b::Float64, a::Vector)
    if a[1] > a[2]
        print(a[1], "\t", a[2], "\n")
        throw("wrong order in vector a")
    end
    res = (b >= a[1]) & (b < a[2])
    return res
end


function gendata(N_g, N_p, break_point, param, pol)
    #N_g = 200
    #N_p = 2000
    #break_point = 10
    #family_pov, vacancy_pov, choice_set = MatchingModel.gen_data(N_g, N_p, break_point)
    dists = rand(truncated(LogNormal(2.75, 0.6), 2, 60), N_g, N_p) ./ 60
    subsidy = rand(Uniform(0,1), N_p) .< .8
    qual = (rand(Uniform(0,1), N_p) .< 0.2)
    home = rand(Uniform(0,1), N_p) .< 0.4

    v_var = rand(Uniform(0,1), N_g) .< 0.5

    family_pov = [dists repeat(subsidy', N_g) repeat(qual', N_g) repeat(home', N_g)]

    vacancy_pov = [dists' repeat(v_var', N_p)]

    choice_set = dists .< break_point

    N_f = rand(1000:2000, N_g)
    N_v = rand(50:100, N_p)
    #β_ft = 2.75
    #αt = [0.5, 0.25, 0.9]
    #At = 0.4
    #β_vt = 3.0
    ##R = 5
    #σt = 2.0


    share_f, share_v = MatchingModel.get_equilibrium(vacancy_pov, family_pov, param, N_f, N_v, choice_set, pol)


    p_match_f, p_match_v = MatchingModel.fx_once(share_f, share_v, N_f, N_v, N_g, N_p, param, choice_set, pol)

    # Check equilibrium
    #Nm_f = share_f .* N_f
    #Nm_v = share_v .* N_v
    #p_match_f[4,1] * Nm_f[4,1]
    #p_match_v[1,4] * Nm_v[1,4]

    index_search_f = Vector{Vector{Float64}}(undef, N_g)
    index_match_f = Vector{Vector{Float64}}(undef, N_g)
    index_search_v = Vector{Vector{Float64}}(undef, N_p)
    index_match_v = Vector{Vector{Float64}}(undef, N_p)
    for i = 1:N_g
        index_search_f[i] = rand(Uniform(0,1), N_f[i])
        index_match_f[i] = rand(Uniform(0,1), N_f[i])
    end
    for i = 1:N_p
        index_search_v[i] = rand(Uniform(0,1), N_v[i])
        index_match_v[i] = rand(Uniform(0,1), N_v[i])
    end

    y_search_f = y_search(share_f, index_search_f, N_f)
    y_match_f = y_match(p_match_f, index_match_f, N_f, y_search_f)
    return y_match_f, vacancy_pov, family_pov, param, N_f, N_v, choice_set
end

end
