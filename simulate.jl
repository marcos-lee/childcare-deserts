provider has 2 qualities high low, 2 modes home and center, 4 types q = 1,2,3,4

10 geographical units, 5_000 families divided across them

500 unique (home, qual, subsidy, dist) vacancies, repeat 20 times to get 10_000 vacancies
note that dist is continuous but will be constrained to be a dummy called far
this results in 8 types of vacancies

each G unit has 50 possibly repeated unique vacancies

include("MatchingModel.jl")
using .MatchingModel

const m = MatchingModel

using Distributions
#using Revise

N_g = 10
N_p = 500
N_v = 10_000
N_f = 5_000

#dist_vacancy = repeat(rand(Uniform(4,35), N_p), 20)
#far = dist_vacancy .> 19.5

dists = rand(truncated(LogNormal(2.75, 0.5), 4, 50), N_g, N_p)
zip_choice_set = dists .< 18
sum(zip_choice_set, dims = 1)
sum(zip_choice_set, dims = 2)
sum(dists .< 12, dims = 2)

subsidy = rand(Uniform(0,1), N_p) .< .8
qual = 1 .+ (rand(Uniform(0,1), N_p) .< 0.2)
home = rand(Uniform(0,1), N_p) .< 0.4


family_pov_temp = Array{Array{Float64}}(undef, N_g)
for i = 1:N_g
    temp = zip_choice_set[i,:]
    family_pov_temp[i] = [i*ones(sum(temp)) subsidy[temp] qual[temp] home[temp] dists[i, temp] (dists[i, temp] .> 12)]
end
family_pov = DataFrame(reduce(vcat, family_pov_temp), [:zipcode, :subsidy, :qual, :home, :dists, :far])
types = unique(family_pov[:, [:subsidy, :qual, :home, :far]])
types[:, :type] = 1:16
family_pov = leftjoin(family_pov, types, on = [:subsidy, :qual, :home, :far])
family_pov = repeat(family_pov, inner = 20)


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


function μ_f(subsidy, qual, home, far, α)
    util = α[1] + α[2] * subsidy + α[3] * qual + α[4] * home + α[5] * far
end

function lnEU_f(subsidy, qual, home, far, p_match, α)
    lnEU_f = μ_f(subsidy, qual, home, far, α) + log(p_match)
end






##############################


a = vacancy_pov_unique[vacancy_pov_unique.id .==1, :]

exp.(lnEU_v.(a.subsidy, a.dists, Ref(p_match), Ref(β), Ref(R)))./sum(exp.(lnEU_v.(a.subsidy, a.dists, Ref(p_match), Ref(β), Ref(R))))
R
log(0.1)
p_match = 0.25
A = .4
β = 2.0
R = 5
