provider has 2 qualities high low, 2 modes home and center, 4 types q = 1,2,3,4

10 geographical units, 10_000 families divided across them

500 unique (home, qual, subsidy, dist) vacancies, repeat 20 times to get 10_000 vacancies

using Distributions, BenchmarkTools, Revise, Random
import NLopt
include("MatchingModel.jl")
using .MatchingModel
include("GenSample.jl")
using .GenSample
include("Estimation.jl")
using .Estimation

#const m = MatchingModel

#using Revise
function gendata()
    N_g = 20
    N_p = 200
    break_point = 10
    family_pov, vacancy_pov, choice_set = MatchingModel.gen_data(N_g, N_p, break_point)
    N_f = rand(1000:2000, N_g)
    N_v = rand(50:100, N_p)
    β_ft = 2.75
    αt = [0.5, 0.25, 0.9]
    At = 0.4
    β_vt = 3.0
    #R = 5
    σt = 2.0
    ct = 1.0/σt
    #α, β_f, β_v, A, c
    param = vcat(αt, log(β_vt-1), log(β_ft-1), log(At / (1-At)), log(ct / (1 - ct)))

    share_f, share_v = MatchingModel.get_equilibrium(vacancy_pov, family_pov, param, N_f, N_v, choice_set)


    p_match_f, p_match_v = MatchingModel.fx_once(share_f, share_v, N_f, N_v, N_g, N_p, param, choice_set)

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

    y_search_f = GenSample.y_search(share_f, index_search_f, N_f)
    y_match_f = GenSample.y_match(p_match_f, index_match_f, N_f, y_search_f)
    return y_match_f, vacancy_pov, family_pov, param, N_f, N_v, choice_set
end

#@btime likelihood(y_match_f, vacancy_pov, family_pov, param, N_f, N_v, choice_set)
#function wraplike(param)
#    return Estimation.likelihood(y_match_f, vacancy_pov, family_pov, param, N_f, N_v, choice_set)
#end

function wraplike1(param::Vector, grad::Vector)
    if length(grad) > 0
        nothing
    end
    test = 0.0
    test = Estimation.likelihood(y_match_f, vacancy_pov, family_pov, param, N_f, N_v, choice_set)
    #global count
    #count::Int += 1
    #println("Iteration: $count; value $test \n; param $param")
    return test
end

#@profiler Estimation.likelihood(y_match_f, vacancy_pov, family_pov, param, N_f, N_v, choice_set)
#@code_warntype wraplike1(param, [])
using Optim

function iters(i)
    function wraplike1(param::Vector, grad::Vector)
        if length(grad) > 0
            nothing
        end
        test = 0.0
        test = Estimation.likelihood(y_match_f, vacancy_pov, family_pov, param, N_f, N_v, choice_set)
        count::Int += 1
        #println("Iteration: $count; value $test \n; param $param")
        return test
    end
    Random.seed!(54321486+i)
    y_match_f, vacancy_pov, family_pov, param, N_f, N_v, choice_set = gendata()
    β_ft = 2.75
    αt = [0.5, 0.25, 0.9]
    At = 0.4
    β_vt = 3.0
    σt = 2.0
    ct = 1.0/σt
    init = vcat(αt, log(β_vt-1), log(β_ft-1), log(At / (1-At)), log(ct / (1 - ct)))
    count = 0
    opt = NLopt.Opt(:LN_NEWUOA, size(init)[1])
    opt.min_objective = wraplike1
    opt.xtol_rel = 1e-12
    opt.ftol_rel = 1e-15
    opt.maxeval = 125_000
    (optf, optx, ret) = NLopt.optimize(opt, init)
    return optx, ret, count
end

numiter = 50
results = Array{Array{Float64,1},1}(undef, numiter)
rets = Array{Any,1}(undef, numiter)
evals = Array{Int64,1}(undef, numiter)
for iter = 1:numiter
    optx, ret, count = iters(iter)
    #count = 0
    #(optf, optx, ret) = NLopt.optimize(opt, init)
    α_e, β_ve, β_fe, A_e, c_e = MatchingModel.unpack(optx)
    results[iter] = vcat(α_e, β_ve, β_fe, A_e, c_e )
    rets[iter] = ret
    evals[iter] = count
end
