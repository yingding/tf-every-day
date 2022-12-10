from datetime import datetime
from pytz import timezone as ptimezone

def get_local_time_str(target_tz_str: str = "Europe/Berlin", format_str: str = "%Y-%m-%d %H-%M-%S") -> str:
    """
    this method is created since the local timezone is miss configured on the server
    @param: target timezone str default "Europe/Berlin"
    @param: "%Y-%m-%d %H-%M-%S" returns 2022-07-07 12-08-45
    """
    target_tz = ptimezone(target_tz_str) # create timezone, in python3.9 use standard lib ZoneInfo
    # utc_dt = datetime.now(datetime.timezone.utc)
    target_dt = datetime.now(target_tz)
    return datetime.strftime(target_dt, format_str)