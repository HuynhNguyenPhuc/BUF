import datetime

def is_weekend(date):
    return date.weekday() in [5, 6]

def is_valid_date(date):
    try:
        datetime.datetime.strptime(date, '%d/%m/%Y')
        return True
    except ValueError:
        return False