import json
import requests
from bs4 import BeautifulSoup
import re
import os
from pathlib import Path
from commonUtil.timeCheck import logging_time

# --- 설정 정보 ---
BASE_URL = "https://km.kyobodts.co.kr"  # 실제 사이트 URL로 변경하세요. (http 또는 https 확인)
LOGIN_URL = f"{BASE_URL}/j_spring_security_check"

YOUR_USERNAME = "jby1303"  # 여기에 실제 로그인 아이디 입력
YOUR_PASSWORD = "dbswlsqo3#"  # 여기에 실제 로그인 비밀번호 입력
# YOUR_PASSWORD = "dbswlsqo22@1"  # 여기에 실제 로그인 비밀번호 입력
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


def makePageNum(pageNum,ctid):
    # board_url = "https://km.kyobodts.co.kr/bbs/bbsFinder.do?method=listView&coid=156&ctid=321" #공지사항 게시판 분해 체크중
    # board_url = " https://km.kyobodts.co.kr/bbs/bbsFinder.do?method=list&coid=156&ctid=321"
    board_url = "https://km.kyobodts.co.kr/bbs/bbsFinder.do?method=list&coid=156&ctid="+ctid+"&page="+str(pageNum)
    board_response = session.get(board_url)
    # print(board_response.content.decode('utf-8'))

    if board_response.status_code == 200:
        data = board_response.json()

        # rows가 존재하는지 확인
        if 'rows' in data and data['rows']:
            doc_numbers = [row['docNumber'] for row in data['rows']]
            # print(f"추출된 docNumber: {doc_numbers}")
            # print(f"총 개수: {len(doc_numbers)}")
        else:
            print("rows 데이터가 없습니다")
            doc_numbers = []
    else:
        print(f"요청 실패: {board_response.status_code}")
        doc_numbers = []

    return doc_numbers

def captBoard(pagenNum,ctid,bbs_Id):
    #board_url = "https://km.kyobodts.co.kr/bbs/bbs.do?method=get&coid=156&ctid=321&bbsId=B0000111&docNumber=12202"  # 예시 게시판 URL, 실제 URL로 변경하세요.
    # board_url = "https://km.kyobodts.co.kr/bbs/bbs.do?method=get&coid=156&ctid=321&bbsId=B0000111&docNumber="+str(pagenNum)  # 예시 게시판 URL, 실제 URL로 변경하세요.
    board_url = "https://km.kyobodts.co.kr/bbs/bbs.do?method=get&coid=156&ctid="+ctid+"&bbsId="+bbs_Id+"&docNumber=" + str(
        pagenNum)  # 예시 게시판 URL, 실제 URL로 변경하세요.
    print(f"\n게시판 페이지 접근 시도: {board_url} ...")
    board_response = session.get(board_url)
    # board_response = requests.get(board_url)
    print(f"게시판 요청 상태 코드: {board_response.status_code}")
    try :
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
            
            #임시주석
            # for key, value in result.items():
            #     if key == '첨부파일':
            #         print(f"\n{key}:")
            #         for idx, file_info in enumerate(value, 1):
            #             print(f"  [{idx}] {file_info.get('파일명', '')} {file_info.get('파일크기', '')}")
            #     elif key == '본문':
            #         print(f"\n{key}:")
            #         print("-" * 50)
            #         print(clean_text(value))
            #         print("-" * 50)
            #     else:
            #         print(f"{key}: {value}")

            return result

        else:
            print("게시판 페이지 접근 실패!")
            print(board_response.text)  # 실패 시 응답 확인

    except requests.exceptions.RequestException as e:
        print(f"요청 중 오류 발생: {e}")

    finally:
        # 세션은 보통 스크립트가 끝날 때 자동으로 닫히지만, 명시적으로 닫을 수도 있습니다.
        #session.close()
        print("\n작업 완료.")

#def loginStart():
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
                print(response.text)
            else:
                print("로그인 성공! (추정)")
                print(f"현재 URL: {response.url}")  # 로그인 성공 후 이동된 URL 확인

                # --- 로그인 성공 후 게시판 URL 호출 (예시) ---
                # 이제 'session' 객체를 사용하여 로그인된 상태로 다른 페이지에 접근할 수 있습니다.
                # board_url = f"{BASE_URL}/board/list.do" # 예시 게시판 URL, 실제 URL로 변경하세요.

                #board_url = "https://km.kyobodts.co.kr/bbs/bbsFinder.do?method=listView&coid=156&ctid=321" #공지사항 게시판 분해 체크중
                #board_url = " https://km.kyobodts.co.kr/bbs/bbsFinder.do?method=list&coid=156&ctid=321"
                #board_response = session.get(board_url)
                #print(board_response.content.decode('utf-8'))

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

#딕셔너리 타입 \n 제겨용
def clean_newlines(data):
    """딕셔너리 내의 모든 \n을 실제 줄바꿈 또는 공백으로 변환"""
    if isinstance(data, dict):
        return {k: clean_newlines(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_newlines(item) for item in data]
    elif isinstance(data, str):
        # 옵션 1: 공백으로 변환 (한 줄로)
        return data.replace('\n', ' ')
        # 옵션 2: 실제 줄바꿈 유지하려면 그냥 data 반환
        # return data
    return data

def makeTxt(cleaned_data, filepath_base):
    """
    파일 경로와 기본 이름(확장자 제외)을 받아,
    파일이 존재할 경우 순번을 붙여 저장하고, 최종 파일 경로를 반환합니다.
    """
    filepath = Path(filepath_base + ".txt")
    counter = 1

    # 파일이 이미 존재하는지 확인하고, 존재하면 순번을 붙여 새로운 파일 경로 생성
    while filepath.exists():
        # 파일 경로를 '/data/공지/공지_2025-11-12_1.txt' 등으로 변경
        filepath = Path(f"{filepath_base}_{counter}.txt")
        counter += 1

    with open(filepath, "w", encoding="utf-8") as f:
        # 파일 저장 시 '게시일시'의 시분초는 파일 이름에 포함하지 않기 위해
        # cleaned_data 자체를 파일에 저장합니다.
        json.dump(cleaned_data, f, indent="\t", ensure_ascii=False)

    print(f"파일 저장 완료: {filepath}")
    return filepath

def login():
    print(f"로그인 시도: {LOGIN_URL} ...")
    result = True
    try:
        response = session.post(LOGIN_URL, data=login_payload)
        # --- 로그인 성공 여부 확인 ---
        # 로그인 성공 여부는 웹사이트마다 다르게 판단해야 합니다.
        # 일반적으로는 HTTP 상태 코드, 특정 리다이렉션, 또는 응답 HTML 내의 특정 텍스트를 확인합니다.

        print(f"로그인 요청 상태 코드: {response.status_code}")

        if response.status_code == 200:
            # 200 OK가 떨어졌다고 무조건 성공은 아닙니다.
            # 로그인 실패 시에도 200 OK를 반환하면서 로그인 페이지로 다시 돌아가는 경우가 많습니다.
            if "로그인이 실패하였습니다" in response.text or "비밀번호가 일치하지 않습니다" in response.text:
                print("로그인 실패: 응답 내용에 실패 메시지가 포함되어 있습니다.")
                # 실패한 경우의 응답 HTML을 확인하여 디버깅하세요.
                # print(response.text)
                result = False
            else:
                print("로그인 성공! (추정)")
                print(f"현재 URL: {response.url}")  # 로그인 성공 후 이동된 URL 확인
                result = True
    except requests.exceptions.RequestException as e:
        print(f"요청 중 오류 발생: {e}")

    return result

@logging_time
def crawllingStart():
    # 게시판 타입
    # 게시판 토탈 페이지
    if(login()!=True):
        return
    bbs_Id = "B0000111"  # 추출하고 싶은 게시판 입력 이후 다른 변수 자동 셋팅됨
    boardTotalPg = 50;
    ctid = "321"  # 공지 기본 id
    ptype = "공지"

    if bbs_Id == "B0000111":
        ptype = "공지"
        # boardTotalPg = 50
        boardTotalPg = 2
        ctid = "321"  # 공지 기본 id
    elif bbs_Id == "B0000002":
        ptype = "회사소식"
        boardTotalPg = 68
        ctid = "172"  # 회사소식 id
    elif bbs_Id == "B0000004":
        ptype = "사우소식"
        boardTotalPg = 36
        ctid = "164"  # 회사소식 id
    else:
        ptype = "공지"

    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / "data" / ptype

    # 폴더 자동 생성
    os.makedirs(data_dir, exist_ok=True)

    print(data_dir)

    # global pdate
    # 게시글 번호 가져오기
    # 임시주석 하단 풀면 정상
    addTot = []
    for i in range(1,boardTotalPg):
     totPg = makePageNum(i,ctid)
     addTot = addTot + totPg
     unique_list = list(set(addTot))
     print(f"총 게시물 번호 개수: {len(unique_list)}")

     # 게시글 크롤링 및 파일 저장
     for doc_number in unique_list:
         data = captBoard(doc_number, ctid, bbs_Id)

         if data and '게시일시' in data:
             # 파일명을 위한 날짜 부분만 추출 (YYYY-MM-DD)
             # 예: '2025-11-12 10:00:00' 에서 '2025-11-12' 추출
             file_date = data['게시일시'][:10]

             # 기본 파일 경로 베이스 생성 (확장자 .txt 제외)
             # 예: /data/공지/공지_2025-11-12
             file_base_name = f"{ptype}_{file_date}"
             filepath_base = data_dir / file_base_name

             cleaned_data = clean_newlines(data)

             # makeTxt 함수에서 중복 처리 및 저장
             makeTxt(cleaned_data, str(filepath_base))


if __name__ == "__main__":
    crawllingStart()


