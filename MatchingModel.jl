module MatchingModel

function μ_f(qual, center, α)
    util = α[1] + α[2] * qual + α[3] * center
end

function matching_function(A, γ, NS_f, NS_v)
    X = A * (NS_f^γ) * (NS_v^(1-γ))
    return min(X, NS_f, NS_v)
end

function p_match(A, γ, NS_f, NS_v)
    P = matching_function(A, γ, NS_f, NS_v) / (NS_f)
end

function lnEU_f(α, qual, center, p_match)
    lnEU_f = μ_f(α, qual, center) + ln(p_match)
end

function cost(dist, β)
    return dist ^ β
end

function lnEU_v(subsidy, dist, p_match, β, R)
    lnEU_v = log(p_match) + R * subsidy - cost(dist, β) / 100
end

function ϕ_v(subsidy_subset, dist_subset, p_match_subset, β, R)
    ϕ_v = exp.(lnEU_v.(subsidy_subset, dist_subset, Ref(p_match), Ref(β), Ref(R)))./sum(exp.(lnEU_v.(subsidy_subset, dist_subset, Ref(p_match), Ref(β), Ref(R))))
end


end
