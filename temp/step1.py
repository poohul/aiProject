import json
import requests
from bs4 import BeautifulSoup
import re

# --- 설정 정보 ---
BASE_URL = "https://km.kyobodts.co.kr"  # 실제 사이트 URL로 변경하세요. (http 또는 https 확인)
LOGIN_URL = f"{BASE_URL}/j_spring_security_check"

YOUR_USERNAME = "jby1303"  # 여기에 실제 로그인 아이디 입력
YOUR_PASSWORD = "dbswlsqo22@"  # 여기에 실제 로그인 비밀번호 입력

# --- requests.Session() 사용 ---
# requests.Session()을 사용하면 로그인 후 생성되는 쿠키(세션 정보)를
# 자동으로 관리해 주어, 로그인 상태를 유지하면서 다른 페이지에 접근할 수 있습니다.
session = requests.Session()

# --- 로그인 페이로드(데이터) 준비 ---
# 폼에서 "name" 속성으로 정의된 필드 이름을 키(Key)로 사용합니다.
login_payload = {
    "j_username": YOUR_USERNAME,
    "j_password": YOUR_PASSWORD
}


def loginStart():
    # --- 로그인 POST 요청 보내기 ---
    print(f"로그인 시도: {LOGIN_URL} ...")
    try:
        response = session.post(LOGIN_URL, data=login_payload)
        # --- 로그인 성공 여부 확인 ---
        # 로그인 성공 여부는 웹사이트마다 다르게 판단해야 합니다.
        # 일반적으로는 HTTP 상태 코드, 특정 리다이렉션, 또는 응답 HTML 내의 특정 텍스트를 확인합니다.

        print(f"로그인 요청 상태 코드: {response.status_code}")

        if response.status_code == 200:
            # 200 OK가 떨어졌다고 무조건 성공은 아닙니다.
            # 로그인 실패 시에도 200 OK를 반환하면서 로그인 페이지로 다시 돌아가는 경우가 많습니다.
            if "로그인 실패" in response.text or "비밀번호가 일치하지 않습니다" in response.text:
                print("로그인 실패: 응답 내용에 실패 메시지가 포함되어 있습니다.")
                # 실패한 경우의 응답 HTML을 확인하여 디버깅하세요.
                # print(response.text)
            else:
                print("로그인 성공! (추정)")
                print(f"현재 URL: {response.url}")  # 로그인 성공 후 이동된 URL 확인

                # --- 로그인 성공 후 게시판 URL 호출 (예시) ---
                # 이제 'session' 객체를 사용하여 로그인된 상태로 다른 페이지에 접근할 수 있습니다.
                # board_url = f"{BASE_URL}/board/list.do" # 예시 게시판 URL, 실제 URL로 변경하세요.
                board_url = "https://km.kyobodts.co.kr/bbs/bbs.do?method=get&coid=156&ctid=321&bbsId=B0000111&docNumber=12202"  # 예시 게시판 URL, 실제 URL로 변경하세요.

                print(f"\n게시판 페이지 접근 시도: {board_url} ...")
                board_response = session.get(board_url)
                # board_response = requests.get(board_url)
                print(f"게시판 요청 상태 코드: {board_response.status_code}")

                if board_response.status_code == 200:
                    print("게시판 페이지 접근 성공!")
                    # 게시판 페이지의 HTML 내용을 출력하거나 BeautifulSoup으로 파싱할 수 있습니다.
                    # print(board_response.text[:1000]) # 처음 1000자만 출력
                    #print(board_response.content.decode('utf-8'))

                    # BeautifulSoup을 사용한 파싱 예시 (BeautifulSoup4 설치 필요: pip install beautifulsoup4)
                    # 컨텐츠 분해
                    result = extract_bbs_content(board_response.content.decode('utf-8'))

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

                    #from bs4 import BeautifulSoup
                    #soup = BeautifulSoup(board_response.text, 'html.parser')
                    # 여기에서 soup 객체를 사용하여 게시글 제목, 내용 등을 추출합니다.
                    # 예:
                    # for title_tag in soup.find_all('a', class_='post-title'):
                    #     print(title_tag.get_text())

                else:
                    print("게시판 페이지 접근 실패!")
                    print(board_response.text)  # 실패 시 응답 확인
        else:
            print(f"로그인 실패: 예상치 못한 HTTP 상태 코드 {response.status_code}")
            print(response.text)  # 실패 시 응답 내용 확인

    except requests.exceptions.RequestException as e:
        print(f"요청 중 오류 발생: {e}")

    finally:
        # 세션은 보통 스크립트가 끝날 때 자동으로 닫히지만, 명시적으로 닫을 수도 있습니다.
        session.close()
        print("\n작업 완료.")

def extract_bbs_content(html_content):
    """
    게시판 HTML에서 주요 정보를 추출하는 함수
    """
    #파일제목용 날짜
    global pdate
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
        pdate = doc_regdate.get('value', '')
        #global fileNm
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


if __name__ == "__main__":
    #게시판 타입
    global ptype
    ptype = "공지"
    #global pdate
    data = loginStart()
    fileNm = ptype + "_" + pdate[:10]+".txt"
    #f2 = open("./data/test.txt"+{fileNm}, "w")
    #f2 = open("C:/Temp/test.txt", "w")
    with  open("C:/Temp/test.txt", "w",encoding="utf-8") as f:
        len =  json.dump(data, f , indent="\t", ensure_ascii=False)
    #len = f2.write(json.dump(data))
    print("maincodes2.txt:총%d바이트크기의파일을생성하였습니다." % len)
    f.close()
    #print(f"파일이름:{fileNm}")
    #print(f"최종결과: {data}")