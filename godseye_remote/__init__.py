"""Code that is SHIPPED to cluster nodes (Ray runtime_env py_modules) and imported there by name.
Standard library only: the remote Python (3.11) and its packages differ from the project's (3.12); nothing from `godseye` may be imported here."""
