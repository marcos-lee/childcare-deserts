module Estimation

include("MatchingModel.jl")
using .MatchingModel

function likelihood(y_match_f, vacancy_pov, family_pov, param, N_f, N_v, choice_set)
    N_g = size(family_pov,1)
    N_p = size(vacancy_pov,1)
    share_f, share_v = MatchingModel.get_equilibrium(vacancy_pov, family_pov, param, N_f, N_v, choice_set)
    p_match_f, p_match_v = MatchingModel.fx_once(share_f, share_v, N_f, N_v, N_g, N_p, param, choice_set)
    like = 0.0
    for i = 1:N_g
        searching = (y_match_f[i][:] .> 0)
        matched = y_match_f[i]#[searching]
        for k = 1:length(searching)
            if searching[k] == 1
                like += (log(share_f[i, y_match_f[i][k]]) + log(p_match_f[i, y_match_f[i][k]]))
            else
                like += log(sum(  share_f[i, choice_set[i,:]] .* (1 .- p_match_f[i, choice_set[i,:]])   ))
            end
        end
        #for k = 1:length(matched)
        #    like += (log(share_f[i, matched[k]]) + log(p_match_f[i, matched[k]]))
        #end
        #like += sum(.!searching) * log(sum(share_f[i, :] .* p_match_f[i, :]))
    end
    return -like
end


end
