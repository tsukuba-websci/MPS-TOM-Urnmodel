using PolyaUrnSimulator
using DynamicNetworkMeasuringTools
using StatsBase

function history2vec(history::Vector{Tuple{Int,Int}}, interval_num::Int)
    tau = div(length(history), interval_num)
    gamma, _ = calc_heaps(history)
    cc = calc_connectedness(history)
    oc, oo, nc, no = cc[:OC], cc[:OO], cc[:NC], cc[:NO]
    c = calc_cluster_coefficient(history)
    y, _ = calc_youth_coefficient(history, interval_num)
    g, _ = calc_ginilike_coefficient(history)
    r = calc_recentness(history, tau)
    h = mean(calc_local_entropy(history, tau))
    return (; gamma, no, nc, oo, oc, c, y, g, r, h)
end