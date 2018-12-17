

class CornacException(Exception):
    """Exception base class to extend from

    """

    pass


class ScoreException(CornacException):
    """Exception raised in score function when facing unknowns

    """

    pass