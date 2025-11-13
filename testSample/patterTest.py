import re


def patternTest(query):
    within_month_pattern = re.search(r"(\d{4})년\s*(\d{1,2})월\s*(?:이내|내|까지)", query)
    # after_year_pattern = re.search(r"(\d{4})년(?:도)?\s*이후", query)

    after_year_pattern = re.search(r"지난\s*(\d+)\s*개월", query)

    return within_month_pattern , after_year_pattern

if __name__ == "__main__":
    timestamp = 1704067200.0
    within_month_pattern, after_year_pattern = patternTest('지난 1개월 이내')
    print(within_month_pattern,after_year_pattern)