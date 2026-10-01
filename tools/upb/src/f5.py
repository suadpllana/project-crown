def discordia_ages(X, sX, Y, sY, rho):
    """Upper and lower concordia-intercept ages of a discordant U-Pb array given in
    Tera-Wasserburg form. See problem.yaml for the output dictionary."""
    w = tera_wasserburg_to_wetherill(X, sX, Y, sY, rho)
    fit = york_fit(w[:, 0], w[:, 1], w[:, 2], w[:, 3], w[:, 4])
    ages = concordia_intercepts(fit["intercept"], fit["slope"])
    if ages.size != 2:
        raise ValueError("the discordia does not cross concordia twice in the age window")
    sig = intercept_age_sigmas(fit["intercept"], fit["slope"], fit["cov"], ages)
    if fit["mswd"] > 1.0:
        sig = sig * np.sqrt(fit["mswd"])
    return {
        "intercept": fit["intercept"],
        "slope": fit["slope"],
        "cov": fit["cov"],
        "mswd": fit["mswd"],
        "n": fit["n"],
        "ages": ages,
        "sigmas": sig,
    }
