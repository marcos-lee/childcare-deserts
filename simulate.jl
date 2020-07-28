provider has 2 qualities high low, 2 modes home and center, 4 types q = 1,2,3,4

10 geographical units, 10_000 families divided across them

500 unique (home, qual, subsidy, dist) vacancies, repeat 20 times to get 10_000 vacancies
note that dist is continuous but will be constrained to be a dummy called far
this results in 8 types of vacancies

each G unit has 50 possibly repeated unique vacancies

include("MatchingModel.jl")
using .MatchingModel

const m = MatchingModel

using Distributions, DataFrames
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
    family_pov_temp[i] = [i*ones(N_p) subsidy qual home dists[i, :] (dists[i, :] .> 15)]
end
family_pov = DataFrame(reduce(vcat, family_pov_temp), [:zipcode, :subsidy, :qual, :home, :dist, :far])
types = unique(family_pov[:, [:subsidy, :qual, :home, :far]])

β_f = 2.5
α = [-4.25, 1.5, 2.25, 1.75]
p_match = 0.25

a = family_pov[family_pov.zipcode .== 1,:]
ϕ_f(a.subsidy, a.qual, a.home, a.dist, p_match, α, β_f)




#types[:, :type] = 1:16
#family_pov = leftjoin(family_pov, types, on = [:subsidy, :qual, :home, :far])
#family_pov = repeat(family_pov, inner = 20)


# create data set form pov of vacancy, each vacancy has 10 rows which has dist to zipcodes and subsidy
# will be then 10 * 500 * 20


vancacy_pov = vcat(dists, subsidy', qual', home')
vacancy_pov_temp = Array{Array{Float64}}(undef, N_p)

for i = 1:N_p
    vacancy_pov_temp[i] = [i*ones(10) 1:10 repeat([subsidy[i]], 10) repeat([qual[i]], 10) repeat([home[i]], 10) dists[:,i]  dists[:,i] .> 12]
end
vacancy_pov = DataFrame(reduce(vcat, vacancy_pov_temp), [:id, :zipcode, :subsidy, :qual, :home, :dists, :far])
vacancy_pov_unique = leftjoin(vacancy_pov, types, on = [:subsidy, :qual, :home, :far])
vacancy_pov_extend = repeat(vacancy_pov, inner = 20)




##############################


a = vacancy_pov_unique[vacancy_pov_unique.id .==1, :]

exp.(lnEU_v.(a.subsidy, a.dists, Ref(p_match), Ref(β), Ref(R)))./sum(exp.(lnEU_v.(a.subsidy, a.dists, Ref(p_match), Ref(β), Ref(R))))
R
log(0.1)
p_match = 0.25
A = .4
β = 2.0
R = 5
