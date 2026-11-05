"""Presentation only: shared DESLIZAMENTO palette and scoped Matplotlib theme."""
from functools import wraps
from threading import RLock
from textwrap import fill

_RENDER_LOCK = RLock()

BACKGROUND = '#f7f9fc'
PANEL = '#f8fafc'
INK = '#263449'
GRID = '#d8e0e9'
GRAIN = '#7d8da1'
RED = '#d93535'
BLUE = '#276bd2'
GREEN = '#14966c'


def measurement_theme(render):
    """Serialize our renderers and restore shared rcParams, even on failure."""
    @wraps(render)
    def themed(*args, **kwargs):
        import matplotlib as mpl
        from cycler import cycler
        with _RENDER_LOCK, mpl.rc_context({
            'figure.facecolor': BACKGROUND, 'axes.facecolor': PANEL,
            'savefig.facecolor': BACKGROUND, 'savefig.edgecolor': BACKGROUND,
            'savefig.transparent': False, 'text.color': INK,
            'axes.edgecolor': INK, 'axes.labelcolor': INK,
            'xtick.color': INK, 'ytick.color': INK,
            'grid.color': GRID, 'grid.alpha': 1.,
            'font.family': ['monospace'], 'mathtext.fontset': 'dejavusans',
            'axes.prop_cycle': cycler(color=[RED, BLUE, GREEN]),
            'legend.facecolor': PANEL, 'legend.edgecolor': GRID,
            'legend.labelcolor': INK,
        }):
            return render(*args, **kwargs)
    return themed


def wrap_metadata(text, width=120):
    """Wrap presentation text without altering scientific result fields."""
    return '\n'.join(fill(line, width=width, break_long_words=False,
                          break_on_hyphens=False)
                     for line in str(text).splitlines())
