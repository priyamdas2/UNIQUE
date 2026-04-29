function val = soft_threshold(z, gamma)

val = sign(z) .* max(abs(z) - gamma, 0);

end
