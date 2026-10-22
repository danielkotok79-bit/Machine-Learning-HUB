"""
mlhub_compat.py – uruchamia aplikację Streamlit z poprawką zgodności dla starszych wersji Streamlit.

Wszystkie 4 aplikacje ML Hub mają w pasku bocznym przełącznik języka:

    st.segmented_control(..., key="w_lang", required=True, ...)

Parametr `required` istnieje dopiero od Streamlit 1.56. Na starszej wersji (np. 1.51 z Anacondy)
kończy się to błędem: TypeError: ... segmented_control() got an unexpected keyword argument 'required'.

Ten moduł przed startem serwera dodaje `required` do st.segmented_control – zachowanie jak w 1.56:
kliknięcie już zaznaczonej opcji niczego nie odznacza. Na Streamlit >= 1.56 nic nie zmienia.
Same aplikacje zostają bez zmian.

Użycie (zamiast `streamlit run`, ten sam Python, w którym jest Streamlit):

    python mlhub_compat.py run app.py --server.port 8502 -- --lang en

Launcher (app.py obok) uruchamia aplikacje właśnie tak i sam też korzysta z install().
"""
from __future__ import annotations

import functools
import inspect
import os
import sys

NATIVE_SINCE = "1.56"  # od tej wersji Streamlit zna segmented_control(required=...)
_MARK = "_mlhub_compat"


def _has_param(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return True  # nie da się sprawdzić → niczego nie zmieniać


def _with_required(signature: inspect.Signature) -> inspect.Signature:
    """Sygnatura z dodatkowym parametrem `required` – żeby kod sprawdzający parametry widział go jak w 1.56."""
    params = list(signature.parameters.values())
    extra = inspect.Parameter("required", inspect.Parameter.KEYWORD_ONLY, default=False, annotation="bool")
    if params and params[-1].kind is inspect.Parameter.VAR_KEYWORD:
        params.insert(len(params) - 1, extra)
    else:
        params.append(extra)
    return signature.replace(parameters=params)


def install() -> bool:
    """Dodaje `required=` do st.segmented_control, jeśli zainstalowany Streamlit go nie zna.

    Zwraca True, gdy poprawka jest aktywna (zainstalowana teraz lub wcześniej), False, gdy nie jest potrzebna
    (Streamlit >= 1.56) albo niemożliwa (Streamlit < 1.40 nie ma segmented_control).
    """
    import streamlit as st
    from streamlit.delta_generator import DeltaGenerator

    original = getattr(DeltaGenerator, "segmented_control", None)
    if original is None:
        return False
    if getattr(original, _MARK, False):
        return True
    if _has_param(original, "required"):
        return False

    @functools.wraps(original)
    def segmented_control(self, label, options, *args, required=False, **kwargs):
        if not required or kwargs.get("selection_mode", "single") != "single":
            return original(self, label, options, *args, **kwargs)  # jak w 1.56: required tylko dla "single"
        state = st.session_state
        key = kwargs.get("key")
        memo = f"_mlhub_last_{key if key is not None else label}"  # ostatnia wybrana opcja

        if key is not None:
            user_cb = kwargs.pop("on_change", None)
            user_args = kwargs.pop("args", None) or ()
            user_kwargs = kwargs.pop("kwargs", None) or {}

            def keep_selection():
                # Callback działa przed kolejnym przebiegiem skryptu – tylko tu wolno zmienić wartość widżetu.
                if state.get(key) is None and state.get(memo) is not None:
                    state[key] = state[memo]  # kliknięto zaznaczoną opcję → zostaje zaznaczona
                    return  # w 1.56 takie kliknięcie też nie wywołuje on_change
                if user_cb is not None:
                    user_cb(*user_args, **user_kwargs)

            kwargs["on_change"] = keep_selection

        value = original(self, label, options, *args, **kwargs)
        if value is None:  # bez klucza nie da się cofnąć kliknięcia – zwróć ostatni wybór
            return state.get(memo)
        state[memo] = value
        return value

    segmented_control.__signature__ = _with_required(inspect.signature(original))
    setattr(segmented_control, _MARK, True)
    DeltaGenerator.segmented_control = segmented_control
    main_dg = getattr(st, "_main", None)
    if main_dg is not None:  # skrót st.segmented_control był związany ze starą metodą
        st.segmented_control = main_dg.segmented_control
    return True


def main() -> int:
    """`python mlhub_compat.py run app.py …` – jak `streamlit run app.py …`, ale z poprawką."""
    import streamlit

    patched = install()
    if patched:
        print(f"[mlhub_compat] Streamlit {streamlit.__version__}: st.segmented_control(required=...) "
              f"uzupełnione (natywnie od {NATIVE_SINCE})", file=sys.stderr, flush=True)
    # Jak `python -m streamlit`: na sys.path katalog roboczy (folder aplikacji) zamiast folderu tego pliku.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != here]
    sys.path.insert(0, os.getcwd())

    from streamlit.web.cli import main as streamlit_main

    sys.argv = ["streamlit", *sys.argv[1:]]
    return streamlit_main(prog_name="streamlit")


if __name__ == "__main__":
    sys.exit(main())
