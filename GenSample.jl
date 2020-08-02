module GenSample

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

end
