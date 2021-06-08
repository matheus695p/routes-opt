import warnings
from argparse import ArgumentParser
warnings.filterwarnings("ignore", category=DeprecationWarning)


def arguments_parser():
    """
    El parser de argumentos de parámetros que hay que setiar para entrenar
    una red deep renewal
    Returns
    -------
    args : argparser
        argparser con todos los parámetros del modelo.
    """
    # argumentos
    parser = ArgumentParser()
    parser.add_argument(
        "-f", "--fff", help="haciendo weon a python", default="1")
    # agregar donde correr y guardar datos
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--min_delta', type=float, default=10e-3)
    parser.add_argument('--optimizer', type=str, default="Adam")
    # parser.add_argument('--loss', type=str,
    #                     default='mean_squared_logarithmic_error')
    parser.add_argument('--loss', type=str,
                        default='mean_squared_error')
    # parser.add_argument('--loss', type=str, default='mean_absolute_error')
    parser.add_argument('--lr_factor', type=float, default=0.75)
    parser.add_argument('--lr_patience', type=int, default=10)
    parser.add_argument('--timesteps', type=int, default=5)
    parser.add_argument('--lr_min', type=float, default=1e-7)
    parser.add_argument('--validation_size', type=float, default=0.2)
    parser.add_argument('--date_format', type=str,
                        default="%Y-%m-%d %H:%M:%S")
    args = parser.parse_args()
    return args
