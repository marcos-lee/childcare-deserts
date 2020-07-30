provider has 2 qualities high low, 2 modes home and center, 4 types q = 1,2,3,4

10 geographical units, 10_000 families divided across them

500 unique (home, qual, subsidy, dist) vacancies, repeat 20 times to get 10_000 vacancies

include("MatchingModel.jl")
using .MatchingModel

const m = MatchingModel

using Distributions, DataFrames, BenchmarkTools, Revise
#using Revise

N_g = 10
N_p = 500
N_v = 10_000
N_f = 10_000


#dist_vacancy = repeat(rand(Uniform(4,35), N_p), 20)
#far = dist_vacancy .> 19.5

dists = rand(truncated(LogNormal(2.75, 0.5), 4, 50), N_g, N_p)
subsidy = rand(Uniform(0,1), N_p) .< .8
qual = 1 .+ (rand(Uniform(0,1), N_p) .< 0.2)
home = rand(Uniform(0,1), N_p) .< 0.4


family_pov_temp = Array{Array{Float64}}(undef, N_g)
for i = 1:N_g
    family_pov_temp[i] = [i*ones(N_p) subsidy qual home dists[i, :] (1/size(subsidy,1))*ones(size(subsidy,1))]
end
family_pov = DataFrame(reduce(vcat, family_pov_temp), [:zipcode, :subsidy, :qual, :home, :dist, :p_match])

# family_pov: there are 10 types of families, searching for 500 types of vacancies


#a = family_pov[family_pov.zipcode .== 1,:]
#aa = MatchingModel.ϕ_f(a.subsidy, a.qual, a.home, a.dist, a.p_match, α, β_f)


#types[:, :type] = 1:16
#family_pov = leftjoin(family_pov, types, on = [:subsidy, :qual, :home, :far])
#family_pov = repeat(family_pov, inner = 20)


# create data set form pov of vacancy, each vacancy has 10 rows which has dist to zipcodes and subsidy
# will be then 10 * 500 * 20


vancacy_pov = vcat(dists, subsidy', qual', home')
vacancy_pov_temp = Array{Array{Float64}}(undef, N_p)

for i = 1:N_p
    vacancy_pov_temp[i] = [i*ones(N_g) 1:N_g subsidy[i]*ones(N_g) qual[i]*ones(N_g) home[i]*ones(N_g) dists[:,i] (1/10)*ones(10)]
end
vacancy_pov = DataFrame(reduce(vcat, vacancy_pov_temp), [:id, :zipcode, :subsidy, :qual, :home, :dist, :p_match])
#vacancy_pov_extend = repeat(vacancy_pov, inner = 20)
vacancy_pov
#vacancy_pov: there are 500 types of vacancies searching for 10 types of families
##############################
#a = vacancy_pov[vacancy_pov.id .==1, :]
#MatchingModel.ϕ_v(a.subsidy, a.dists, a.p_match, β_v, R)

β_f = 2.9
α = [-4.25, 1.5, 2.25, 1.75]
A = .4
β_v = 2.0
R = 5
γ = 0.4
param = vcat(α, β_v, β_f, R, A, γ)

include("MatchingModel.jl")
using .MatchingModel
share_f, share_v = MatchingModel.get_equilibrium(vacancy_pov, family_pov, param)

sort(share_f, dims = 2) .* 1000
#MatchingModel.matching_function(A, γ, NS_f[1][1], NS_v[1][1]) / NS_f[1][1]
