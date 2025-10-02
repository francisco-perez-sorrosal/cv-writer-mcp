"""CV Writer MCP Server - Convert markdown CV content to LaTeX and compile to PDF."""

from .conversion import MD2LaTeXAgent

__version__ = "0.1.0"
__author__ = "Francisco Perez-Sorrosal"
__email__ = "fperezsorrosal@gmail.com"

__all__ = [
    "MD2LaTeXAgent",
]
