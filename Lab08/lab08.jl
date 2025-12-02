# Zadania na dzisiaj:
# 9.1 - 9.6

# problem 9.1
##
# firwin_lp_I
function firwin_lp_I(order, F0)
    L = div(order, 2)
    st = -(((N+1)-1)/2)
    en = ((N+1)-1)/2
    for m in st:en
        if m == 0
            return 2 * F0
        else
            return 2 * F0 * ((sin(2π*F0*m))/2π*F0*m)
        end
    end
end

# problem 9.2
##
# firwin_hp_I
function firwin_hp_I(order, F0)
    L = div(order, 2)
    st = -(((N+1)-1)/2)
    en = ((N+1)-1)/2
    for m in st:en
        if m == 0
            return 1 - 2*F0
        else
            return -2*F0*(sin(2π*F0*m)/2π*F0*m)
        end
    end
end

# problem 9.3
##
# firwin_bp_I
function firwin_bp_I(order, F1, F2)
    st = -(((N+1)-1)/2)
    en = ((N+1)-1)/2
    if order < F1
        for m in st:en
            if m == 0
                return 2 * F0
            else
                return 2 * F0 * ((sin(2π*F0*m))/2π*F0*m)
            end
        end
    elseif order > F2
        for m in st:en
            if m == 0
                return 1 - 2*F0
            else
                return -2*F0*(sin(2π*F0*m)/2π*F0*m)
            end
        end
    end
end

# problem 9.4
##
# firwin_bs_I
function firwin_bs_I(order, F1, F2)
    h_lp_f1 = firwin_lp_I(order, F1)
    h_hp_f2 = firwin_hp_I(order, F2)

    h_bs = zeros(Float64, order + 1)

    for n in 1:(order+1)
        h_bs[n] = h_lp_f1[n] + h_hp_f2[n]
    end

    return h_bs
end

# problem 9.5
##
# firwin_lp_II
function firwin_lp_II(order, F0)
    L = div(order, 2)
    h = zeros(Float64, order)

    for m in 1:order-1
        if m == L
            h[m+1] = 2 * F0
        else
            h[m+1] = sin(2*π*F0*(m-L)) / (π*(m-L))
        end
    end

    return h
end

# problem 9.6
##
# firwin_bp_II
function firwin_hp_II(order, F0)
    h_lp = firwin_lp_II(order, F0)
    N = order
    h_hp = zeros(Float64, order)

    for n in 1:order
        if n == div(order, 2) + 1
            h_hp[n] = 1.0 - h_lp[n]
        else
            h_hp[n] = -h_lp[n]
        end
    end
    
    return h_hp
end

function firwin_bp_II(order, F1, F2)
    h_lp_f2 = firwin_lp_II(order, F2)
    h_hp_f1 = firwin_hp_II(order, F1)

    h_bp = zeros(Float64, order)

    for n in 1:order
        h_bp[n] = h_lp_f2[n] - h_hp_f1[n]
    end

    return h_bp
end