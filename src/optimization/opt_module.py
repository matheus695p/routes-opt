
def lower_limit_constraint(model):
    """
    poner la condicción en el limite para la variable routes

    Parameters
    ----------
    model : pyomo.model
        modelo de pyomo.

    Returns
    -------
    TYPE
        restricción para agregar.

    """
    return model.routes >= 0


def upper_limit_constraint(model):
    """
    poner la condicción en el limite para la variable routes

    Parameters
    ----------
    model : pyomo.model
        modelo de pyomo.

    Returns
    -------
    TYPE
        restricción para agregar.

    """
    return model.routes <= 0
