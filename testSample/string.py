
# ptype = "공지"
# pdate = "2025-05-27 11:33:57"
# #print(pdate[:10])
# tempvale = ptype + "_" + pdate[:10]
# print(tempvale)

pagenNum = 1
board_url = "https://km.kyobodts.co.kr/bbs/bbs.do?method=get&coid=156&ctid=321&bbsId=B0000111&docNumber=" + str(pagenNum)  # 예시 게시판 URL, 실제 URL로 변경하세요.
print(board_url)