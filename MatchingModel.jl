module MatchingModel

function matching_function(A, γ, NS_f, NS_v)
    X = A * (NS_f^γ) * (NS_v^(1-γ))
    return min(X, NS_f, NS_v)
end

function p_match(A, γ, NS_f, NS_v)
    P = matching_function(A, γ, NS_f, NS_v) / (NS_f)
end

function μ_f(subsidy, qual, home, dist, α, β_f)
    util = α[1] + α[2] * subsidy + α[3] * qual + α[4] * home - cost(dist, β_f) /2500
end

function lnEU_f(subsidy, qual, home, dist, p_match, α, β_f)
    lnEU_f = μ_f(subsidy, qual, home, dist, α, β_f) + log(p_match)
end

function cost(dist, β)
    return dist ^ β
end

function lnEU_v(subsidy, dist, p_match, β, R)
    lnEU_v = log(p_match) + R * subsidy - cost(dist, β) / 100
end

function ϕ_v(subsidy_subset, dist_subset, p_match, β_v, R)
    temp = lnEU_v.(subsidy_subset, dist_subset, p_match, Ref(β_v), R)
    ϕ_v = exp.(temp) ./ sum(exp.(temp))
    return ϕ_v
end

function ϕ_f(subsidy_subset, qual_subset, home_subset, dist_subset, p_match, α, β_f)
    temp = lnEU_f.(subsidy_subset, qual_subset, home_subset, dist_subset, p_match, Ref(α), β_f)
    ϕ_f = exp.(temp) ./ sum(exp.(temp))
    return ϕ_f
end

end
