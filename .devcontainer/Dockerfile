# 원하는 파이썬 버전 (예: 3.9, 3.10, 3.11 등)
FROM python:3.10-slim

# 기본 업데이트 및 git 설치 (협업/버전관리를 위해 보통 필요)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /workspaces/Coexistence/simulation

# (참고) requirements.txt 설치는 devcontainer.json의 postCreateCommand가 담당합니다.