using Distributions, BenchmarkTools, Revise, Random
import NLopt
using ForwardDiff, FiniteDiff, LinearAlgebra
include("MatchingModel.jl")
using .MatchingModel
include("GenSample.jl")
using .GenSample
include("Estimation.jl")
using .Estimation

N_g = 50
N_p = 500
break_point = 20 /60
αt = [0.5, 0.25, 0.9]
#β_vt = [2.1, 0.07]
#β_vt = 0.07
#β_ft = [2.5, 0.05]
β_vt = 2.7
β_ft = 3.0
At = 0.4
σt = 2.0
γt =1.2
ct = 1.0/σt
#α, β_f, β_v, A, c, , 
param = vcat(αt, log(β_vt-1), log(At / (1-At)), log(ct / (1-ct)), γt)

pol = 0
Random.seed!(410)
y_match_f, vacancy_pov, family_pov, param, N_f, N_v, choice_set = GenSample.gendata(N_g, N_p, break_point, param, pol)

function wraplike(param::Vector)
    return Estimation.likelihood(y_match_f, vacancy_pov, family_pov, param, N_f, N_v, choice_set, pol)
end
function wraptest!(param::Vector, grad::Vector)
    if length(grad) > 0
        grad[:] = ForwardDiff.gradient(wraplike, param)
        #grad[:] = ForwardDiff.gradient(wraplike, x)
        if iszero(maximum(isnan.(grad)))
            grad_max_val = findmax(abs.(grad))[1]
            grad_max_ind = findmax(abs.(grad))[2]
            @time grad_norm = norm(grad)
            println("Maximum gradient at $grad_max_ind with value $grad_max_val and norm $grad_norm.")
        else
            println("Gradient has NaN elements.")
        end
    end
    global counter
    counter += 1
    test = wraplike(param)
    println("Iteration: $counter; value $test \n; param $param")
    return test
end

init = vcat(αt, log(β_vt-1), log(At / (1-At)), log(ct / (1-ct)), γt)

nlopt = NLopt.Opt(:LN_SBPLX, size(init)[1])
nlopt = NLopt.Opt(:LD_LBFGS, size(init)[1])
#nlopt = NLopt.Opt(:LN_SBPLX, size(param)[1])
nlopt.min_objective = wraptest!
nlopt.maxeval = 1000
counter = 0
(optf, optx, ret) = NLopt.optimize(nlopt, init)
init = copy(optx)
MatchingModel.unpack(optx,pol)

stderr = sqrt.(diag(inv(ForwardDiff.hessian(wraplike, optx))))

optx ./ stderr

wraptest!(param, [])

(32310.437668752948, [0.2637043667890306, 0.3933628938563233, 0.6096712681996171, 2.524312502067534, 0.09459465785644514, -0.41762939581101716, -0.3891513556532999, 6.178186404624114], :SUCCESS)

wraptest1(ones(Dual, 7) .+ Dual(0,2.1))
init = vcat(opt.minimizer[1:3], opt.minimizer[4:5] ./2 , log(0.9 / (1-0.9)), log(0.1 / (1 - 0.1)))

init = vcat(ones(3), log(1), log(1), log(0.9 / (1-0.9)), log(0.1 / (1 - 0.1)))
func = OnceDifferentiable(wraptest1, init; autodiff = :central)
opt = Optim.optimize(func, init, LBFGS(linesearch=LineSearches.BackTracking(),
                           alphaguess=LineSearches.InitialStatic(scaled=true, alpha=0.1)),
                            Optim.Options(show_trace=true, allow_f_increases=true))
MatchingModel.unpack(opt.minimizer)
func1 = TwiceDifferentiable(wraptest1, init; autodiff = :central)
opt = Optim.optimize(func1, init,
                            Optim.Options(show_trace=true, allow_f_increases=true))

a, b, c, d, e = MatchingModel.unpack(opt.minimizer)
parameters = vcat(a,b,c,d,e)

numerical_hessian = hessian!(func1,opt.minimizer)
var_cov_matrix = inv(numerical_hessian)
temp = sqrt.(diag(var_cov_matrix))

t_stats = parameters ./ temp
function wraptest1(param::Vector)
    Estimation.likelihood(y_match_f, vacancy_pov, family_pov, param, N_f, N_v, choice_set)
end
#rets = Array{Char,1}(undef, numiter)
#for i = 1:numiter
#    rets[i,:] = results[i]
#end
writedlm("Full_Large_sample_results.csv", results, ",")
writedlm("Full_Large_sample_results_evals.csv", evals, ",")
writedlm("Full_Large_sample_results_rets.csv", rets, ",")

#writedlm("MC_results_rets.txt", rets, ",")
MatchingModel.unpack(results[1,:])
x = Dual(2, 1)



Random.seed!(54321486+1)
y_match_f, vacancy_pov, family_pov, param, N_f, N_v, choice_set = gendata()
share_f, share_v = MatchingModel.get_equilibrium(vacancy_pov, family_pov, param, N_f, N_v, choice_set)
p_match_f, p_match_v = MatchingModel.fx_once(share_f, share_v, N_f, N_v, 200, 2000, param, choice_set)

Nm_f = share_f .* N_f
Nm_v = share_v .* N_v

a = p_match_f .* Nm_f
b = p_match_v .* Nm_v


#MatchingModel.cost(18, [4.5, -1.3, 0.09])
#MatchingModel.cost(5, β_ft)
numiter = 1
results = Array{Float64,2}(undef, numiter, 10)
rets = Array{String,1}(undef, numiter)
evals = Array{Int64,1}(undef, numiter)
for iter = 1:numiter
    print(iter)
    optx, ret, count = iters(iter, N_g, N_p, break_point, αt, β_ft, β_vt, At, σt)
    #count = 0
    #(optf, optx, ret) = NLopt.optimize(opt, init)
    α_e, β_ve, β_fe, A_e, c_e = MatchingModel.unpack(optx)
    results[iter,:] = vcat(α_e, β_ve, β_fe, A_e, c_e )
    rets[iter] = String(ret)
    evals[iter] = count
end

function gendata()
    N_g = 200
    N_p = 2000
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

using Optim, LineSearches

function iters(i, N_g, N_p, break_point, αt, β_ft, β_vt, At, σt)
    function wraptest(param::Vector)
        Estimation.likelihood(y_match_f, vacancy_pov, family_pov, param, N_f, N_v, choice_set)
    end
    function wraplike_logit(param::Vector, grad::Vector)
        if length(grad) > 0
            nothing
        end
        test = 0.0
        test = Estimation.likelihood_logit(y_match_f, vacancy_pov, family_pov, param, N_f, N_v, choice_set)
        count::Int += 1
        println("Iteration: $count; value $test \n; param $param")
        return test
    end
    Random.seed!(54321486+i)
    y_match_f, vacancy_pov, family_pov, param, N_f, N_v, choice_set = GenSample.gendata(N_g, N_p, break_point, αt, β_ft, β_vt, At, σt)
    count = 0
    opt = NLopt.Opt(:LN_NEWUOA, 5)
    opt.min_objective = wraplike_logit
    opt.xtol_rel = 1e-12
    opt.ftol_rel = 1e-15
    opt.maxeval = 125_000
    (optf, optx, ret) = NLopt.optimize(opt, ones(5)/2)

    #ct = 1.0/σtLN_NEWUOA
    #init = vcat(αt, log(β_vt-1), log(β_ft-1), log(At / (1-At)), log(ct / (1 - ct)))
    init = vcat(optx[1:3], optx[4:5] ./2 , log(0.9 / (1-0.9)), log(0.1 / (1 - 0.1)))
    count = 0
    opt = NLopt.Opt(:LN_NEWUOA, size(init)[1])
    opt.min_objective = wraptest
    opt.xtol_rel = 1e-12
    opt.ftol_rel = 1e-15
    opt.maxeval = 125_000
    (optf, optx, ret) = NLopt.optimize(opt, init)
    return optx, ret, count
end

Estimation.likelihood(y_match_f, vacancy_pov, family_pov, param, N_f, N_v, choice_set, pol)
func = OnceDifferentiable(wraplike, param; autodiff = :forward)
result = Optim.optimize(wraplike, param, Optim.Options(show_trace=true, allow_f_increases=true))
opt = Optim.optimize(func, param, LBFGS(linesearch=LineSearches.BackTracking(),
                    alphaguess=LineSearches.InitialStatic(scaled=true, alpha=0.1)),
                    Optim.Options(show_trace=true, allow_f_increases=true))
MatchingModel.unpack(opt.minimizer)
func1 = TwiceDifferentiable(wraptest, param; autodiff = :forward)
opt = Optim.optimize(func1, param, Optim.Options(show_trace=true, allow_f_increases=true, iterations = 1000))









zij = 2.5
σ = 8
β1 = 5
β2 = 6
α1 = 2
α2 = 3
xz = [rand(1000) rand(1000)]
xp = [rand(1000) rand(1000)]
wzp = [rand(1000) rand(1000)]
pz = [rand(1000) rand(1000)]
pp = [rand(1000) rand(1000)]
function testing1(α1 ,β1, σ)
    @. exp((α1 .* xz[:,1] .+ β1 .* wzp[:,1] .+ pz[:,1]) ./ σ) ./ (exp((α1 .* xz[:,1] .+ β1 .* wzp[1] .+ pz[:,1]) ./ σ) .+ exp((α1 .* xz[:,2] .+ β1 .* wzp[:,2] .- pz[:,2]) ./ σ))
end


function testing2(α2, β2, σ)
    @. exp((α2 * xp[:,1] + β2 * wzp[:,1] .+ pp[:,1]) / σ) ./ (exp((α2 * xp[:,1] + β2 * wzp[:,1] + pp[:,1]) / σ) + exp((α2 * xp[:,2] + β2 * wzp[:,2] + pp[:,2]) / σ))
end

testing1(α1, β1, σ)
testing2(α2, β2, σ)

function tester(α1, α2, β1, β2, σ)
    testing1(α1, β1, σ) / testing2(α2, β2, σ)
end
tester(α1, α2, β1 * 2, 5.0, 7.53)
tester(α1, α2, β1, β2, σ)
function like(θ)
    β1est = θ[1]
    β2est = θ[2]
    return norm(tester(2*α1, 2*α2, β1est, β2est, 2*σ) - tester(α1, α2, β1, β2, σ))
end

function like1(θ)
    β1est = θ[1]
    return norm(tester(2*α1, 2*α2, β1est, β1est, 2*σ) - tester(α1, α2, β1, β1, σ))
end


opt = NLopt.Opt(:LN_BOBYQA, 2)
opt.min_objective = like
opt.xtol_rel = 1e-12
opt.ftol_rel = 1e-15
opt.maxeval = 10000
(optf, optx, ret) = NLopt.optimize(opt, ones(2))
import Optim
Optim.optimize(like, ones(2) .* 10)
Optim.optimize(like1, -100.0, 100.0)

[5.96e+00, 1.03e+01]