provider has 2 qualities high low, 2 modes home and center, 4 types q = 1,2,3,4

10 geographical units, 10_000 families divided across them

500 unique (home, qual, subsidy, dist) vacancies, repeat 20 times to get 10_000 vacancies

include("MatchingModel.jl")
using .MatchingModel

const m = MatchingModel

using Distributions, DataFrames
#using Revise

N_g = 10
N_p = 500
N_v = 10_000
N_f = 10_000

β_f = 2.5
α = [-4.25, 1.5, 2.25, 1.75]

A = .4
β_v = 2.0
R = 5

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


a = family_pov[family_pov.zipcode .== 1,:]
aa = MatchingModel.ϕ_f(a.subsidy, a.qual, a.home, a.dist, a.p_match, α, β_f)


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
a = vacancy_pov[vacancy_pov.id .==1, :]
MatchingModel.ϕ_v(a.subsidy, a.dists, a.p_match, β_v, R)

β_f = 2.5
α = [-4.25, 1.5, 2.25, 1.75]
A = .4
β_v = 2.0
R = 5
γ = 0.6
param = [α, β_v, β_f, R, A, γ]
function unpack(param)
    α = param[1]
    β_v = param[2]
    β_f = param[3]
    R = param[4]
    #A = param[5]
    #γ = param[6]
    return α, β_f, β_v, R #,A, γ
end
p_match_f = Array{Array{Float64}}(undef, 10) #will contain 10 arrays of 500x1
p_match_v = Array{Array{Float64}}(undef, 500) #will contain 500 arrays of 10x1
N_f = Array{Float64}(undef, 10) #will contain 10 arrays of 500x1
N_v = Array{Float64}(undef, 500) #will contain 500 arrays of 10x1
for i = 1:10
    p_match_f[i] = (1/500)*ones(500)
    N_f[i] = 1_000
end
for i = 1:500
    p_match_v[i] = (1/10)*ones(10)
    N_v[i] = 20
end

function gen_shares(vacancy_pov, family_pov, p_match_f, p_match_v, param)
    α, β_v, β_f, R = unpack(param)
    share_f = Array{Array{Float64}}(undef, 10) #will contain 10 arrays of 500x1
    share_v = Array{Array{Float64}}(undef, 500) #will contain 500 arrays of 10x1
    for i = 1:10
        temp = family_pov[family_pov.zipcode .== i,:]
        share_f[i] = MatchingModel.ϕ_f(temp.subsidy, temp.qual, temp.home, temp.dist, p_match_f[i], α, β_f)
    end
    for i = 1:500
        temp = vacancy_pov[vacancy_pov.id .== i,:]
        share_v[i] = MatchingModel.ϕ_v(temp.subsidy, temp.dist, p_match_v[i], β_v, R)
    end
    return share_f, share_v
end

share_f, share_v = gen_shares(vacancy_pov, family_pov, p_match_f, p_match_v, param)
NS_v = share_v .* N_v
NS_f = share_f .* N_f

matches = Array{Float64}(undef, 10*500)
p_match_f_new = deepcopy(p_match_f)
p_match_v_new = deepcopy(p_match_v)
p_match_f[1]
for f = 1:10
    for v = 1:500
        p_match_f_new[f][v] = MatchingModel.matching_function(A, γ, NS_f[f][v], NS_v[v][f]) / NS_f[f][v]
        p_match_v_new[v][f] = MatchingModel.matching_function(A, γ, NS_f[f][v], NS_v[v][f]) / NS_v[v][f]
        #matches[i] = MatchingModel.matching_function(A, γ, NS_f[f][v], NS_v[v][f])
    end
end
#MatchingModel.matching_function(A, γ, NS_f[1][1], NS_v[1][1]) / NS_f[1][1]
share_f, share_v = gen_shares(vacancy_pov, family_pov, p_match_f_new, p_match_v_new, param)
NS_v = share_v .* N_v
NS_f = share_f .* N_f
