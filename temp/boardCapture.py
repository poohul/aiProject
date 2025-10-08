from bs4 import BeautifulSoup
import re
import requests

# --- 설정 정보 ---
BASE_URL = "https://km.kyobodts.co.kr" # 실제 사이트 URL로 변경하세요. (http 또는 https 확인)
LOGIN_URL = f"{BASE_URL}/j_spring_security_check"

YOUR_USERNAME = "jby1303"  # 여기에 실제 로그인 아이디 입력
YOUR_PASSWORD = "dbswlsqo22@" # 여기에 실제 로그인 비밀번호 입력

# --- requests.Session() 사용 ---
# requests.Session()을 사용하면 로그인 후 생성되는 쿠키(세션 정보)를
# 자동으로 관리해 주어, 로그인 상태를 유지하면서 다른 페이지에 접근할 수 있습니다.
session = requests.Session()


def extract_bbs_content(html_content):
    """
    게시판 HTML에서 주요 정보를 추출하는 함수
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # 추출할 데이터를 저장할 딕셔너리
    bbs_data = {}

    # 1. 게시자 정보 추출
    user_name_span = soup.find('span', id='userName')
    if user_name_span:
        bbs_data['게시자'] = user_name_span.text.strip()

    # 2. 게시일시 추출
    doc_regdate = soup.find('input', {'id': 'docRegdate'})
    if doc_regdate:
        bbs_data['게시일시'] = doc_regdate.get('value', '')

    # 3. 제목 추출
    doc_subject = soup.find('textarea', {'id': 'docSubject'})
    if doc_subject:
        bbs_data['제목'] = doc_subject.text.strip()

    # 4. 본문 내용 추출 (두 가지 방법 시도)
    # 방법 1: bbsCon div에서 추출
    content_div = soup.find('div', class_='bbsCon')
    if content_div:
        # HTML 태그 제거하고 텍스트만 추출
        content_text = content_div.get_text(separator='\n', strip=True)
        bbs_data['본문'] = content_text

    # 방법 2: textarea에서 추출 (백업)
    if not bbs_data.get('본문'):
        editor_textarea = soup.find('textarea', {'id': 'editor_se'})
        if editor_textarea:
            # HTML 태그 제거
            temp_soup = BeautifulSoup(editor_textarea.text, 'html.parser')
            bbs_data['본문'] = temp_soup.get_text(separator='\n', strip=True)

    # 5. 첨부파일 정보 추출
    file_list = soup.find('ul', {'id': 'ulFileList'})
    if file_list:
        files = []
        for li in file_list.find_all('li', id='liFileList'):
            file_info = {}

            # 파일명
            logical_filename = li.find('input', {'id': 'logicalFileName'})
            if logical_filename:
                file_info['파일명'] = logical_filename.get('value', '')

            # 파일크기
            file_size = li.find('input', {'id': 'fileSize'})
            if file_size:
                file_info['파일크기'] = file_size.get('value', '')

            if file_info:
                files.append(file_info)

        bbs_data['첨부파일'] = files

    return bbs_data


def clean_text(text):
    """
    텍스트에서 불필요한 공백과 특수문자 정리
    """
    # 연속된 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    # 앞뒤 공백 제거
    text = text.strip()
    return text


# 파일에서 HTML 읽기
def extract_from_file(file_path):
    """
    파일에서 HTML을 읽어 내용을 추출
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    result = extract_bbs_content(html_content)

    # 결과 출력
    print("=" * 50)
    print("게시판 내용 추출 결과")
    print("=" * 50)

    for key, value in result.items():
        if key == '첨부파일':
            print(f"\n{key}:")
            for idx, file_info in enumerate(value, 1):
                print(f"  [{idx}] {file_info.get('파일명', '')} {file_info.get('파일크기', '')}")
        elif key == '본문':
            print(f"\n{key}:")
            print("-" * 50)
            print(clean_text(value))
            print("-" * 50)
        else:
            print(f"{key}: {value}")

    return result


# 실행 예시
if __name__ == "__main__":
    # 파일 경로 지정
    file_path = "추출.txt"  # 또는 "대상.txt"

    try:
        data = extract_from_file(file_path)

        # 필요시 JSON으로 저장
        import json

        with open('extracted_data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("\n결과가 'extracted_data.json' 파일로 저장되었습니다.")

    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
    except Exception as e:
        print(f"오류 발생: {e}")