from datetime import datetime, timezone

def conv_timestamp(timestamp):
    date_str = '알 수 없음'
    if isinstance(timestamp, (int, float)) and timestamp > 0:
        try:
            # 타임스탬프를 datetime 객체로 변환하고 원하는 형식으로 포맷팅
            date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        except Exception:
            date_str = '변환 오류'
    return date_str

if __name__ == "__main__":
    timestamp = 1704067200.0
    conv = conv_timestamp(timestamp)
    print(conv)