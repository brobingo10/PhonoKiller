"""Domain-specific exceptions raised by PhonoKiller."""


class PhonoKillerError(RuntimeError):
    """Base class for user-facing workflow errors."""


class StructureValidationError(PhonoKillerError):
    """The input cannot be used as a three-dimensional periodic crystal."""


class CalculatorValidationError(PhonoKillerError):
    """The calculator is invalid or cannot provide a required property."""


class RelaxationError(PhonoKillerError):
    """The relaxation failed or did not converge."""


class DisplacementError(PhonoKillerError):
    """A displaced-supercell calculation failed validation."""


class SoftModeError(PhonoKillerError):
    """Soft-mode analysis or distortion generation failed validation."""


class ResumeMismatchError(PhonoKillerError):
    """Existing output belongs to a different effective run."""


class OutputDirectoryError(PhonoKillerError):
    """The output directory cannot safely be used."""


class CandidateReductionError(PhonoKillerError):
    """A fatal error prevented a candidate batch from being processed."""


class CandidateLimitError(SoftModeError):
    """The requested exhaustive distortion set exceeds the configured limit."""
