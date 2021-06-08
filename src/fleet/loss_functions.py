# import numpy as np
import tensorflow as tf
import keras.backend as K


def handler_loss_function():
    """
    Función de costos para balancear aprendizaje de los pesos de la red
    dado que prefiere una clase mejor que la otra
    Parameters
    ----------
    factor : float, optional
        factor por el cual multiplicar la clase de
        % de cu en cola final. The default is 1.5.
    Returns
    -------
    flotation_loss_function
        DESCRIPTION.
    """
    # Retorna la función de costos de cosmos con penalización
    def route_optimzation_loss(y_true, y_pred):
        """
        Función de costos implementación, el error cuadratico medio
        pero tiene factores multiplicativos en la clase perjudicada por el
        rango de su variable target
        Parameters
        ----------
        y_true : array
            observaciones.
        y_pred : TYPE
            predicciones.
        Returns
        -------
        loss : function
            función de costos.
        """
        # y_true = y_test.copy()
        # y_pred = y_test + 10
        # Covertir a tensor de tensorflow con keras como backend
        y_true = K.cast(y_true, dtype='float32')
        y_pred = K.cast(y_pred, dtype='float32')
        # Reshape como vector
        y_true = K.reshape(y_true, -1)
        y_pred = K.reshape(y_pred, -1)

        loss = K.square(K.log(y_pred + 1) - K.log(y_true + 1))

        loss = K.mean(loss, axis=-1)

        return loss

    return route_optimzation_loss


def handler_routes(factor=1):

    # Retorna la función de costos para penalizar errores por abajo
    def routes_loss_function(y_true, y_pred):
        # Covertir a tensor de tensorflow con keras como backend
        y_true = K.cast(y_true, dtype='float32')
        y_pred = K.cast(y_pred, dtype='float32')
        # Reshape como vector
        y_true = K.reshape(y_true, (-1, 1))
        y_pred = K.reshape(y_pred, (-1, 1))
        # Vector de error mae
        diff_error = y_pred - y_true
        # Cuenta el número de veces que se equivoca por abajo
        negative_values = K.cast(tf.math.count_nonzero(diff_error < 0),
                                 dtype='float32')
        positive_values = K.cast(tf.math.count_nonzero(diff_error >= 0),
                                 dtype='float32')

        loss = K.square(y_pred - y_true)
        loss = K.sum(loss, axis=1)
        loss = K.mean(loss)
        loss = loss * (factor + negative_values / positive_values)
        return loss

    return routes_loss_function
