# 웹 배포 가이드 (Cloudflare + 컨테이너 백엔드 + AdSense)

이 앱은 두 부분으로 배포합니다.

- **프론트엔드**(정적) → Cloudflare Pages
- **백엔드**(FastAPI + ffmpeg + yt-dlp + numpy) → 컨테이너 호스트

> ⚠️ **왜 백엔드를 따로 두나요?**
> 백엔드는 `ffmpeg`(외부 프로세스)와 `yt-dlp`, `numpy` 오디오 분석을 씁니다.
> 이건 **Cloudflare Workers나 Supabase/Firebase 서버리스 함수에서 실행 불가**합니다.
> 반드시 실제 컨테이너/VM이 필요합니다. BaaS는 (원한다면) 로그인·채보 저장 같은
> **부가기능**에만 보조로 쓰세요. 오디오 처리 자체를 대체하지 못합니다.

---

## 선택 A — 자체 호스팅 (가장 간단, Cloudflare 선택)

VPS 한 대만 있으면 됩니다. Docker만 설치하면 한 줄로 뜹니다.

```bash
# 광고 없이
docker compose -f docker-compose.prod.yml up --build -d

# 광고 켜서 (본인 AdSense 값)
VITE_ADSENSE_CLIENT=ca-pub-1234567890123456 VITE_ADSENSE_SLOT=1234567890 \
  docker compose -f docker-compose.prod.yml up --build -d
```

- 브라우저에서 `http://<서버IP>:8080` 접속
- nginx가 정적 프론트를 서빙하고 `/api/*`를 백엔드로 프록시합니다.
- **Cloudflare는 이 서버 앞에 DNS/프록시(주황 구름)로 붙여** TLS·CDN·DDoS 방어를
  얻습니다. Cloudflare 대시보드에서 도메인 A레코드를 서버 IP로 지정하고
  프록시를 켜면 끝입니다. (원하면 8080 대신 80/443으로 매핑하세요.)

## 선택 B — Cloudflare Pages + 별도 백엔드 호스트

프론트는 Cloudflare Pages, 백엔드는 컨테이너 PaaS(Render/Railway/Fly.io 등)나
Cloudflare Containers에 올립니다.

### 1) 백엔드 배포 (예: Render)

- 새 **Web Service** → 이 저장소의 `backend/` 를 지정, `Dockerfile` 사용.
- 호스트가 주입하는 `$PORT`를 자동으로 씁니다(Dockerfile 반영 완료).
- 배포 후 URL 확인: 예) `https://rhythm-backend.onrender.com`
- 헬스체크: `GET /health` → `{"status":"ok"}`

> 주의: 생성된 오디오/채보는 인스턴스 로컬 디스크에 잠깐 저장됩니다.
> **인스턴스 1개(수직 확장)** 로 두세요. 여러 인스턴스로 스케일하면 재생성/오디오
> 재생이 다른 인스턴스로 가서 깨질 수 있습니다.

### 2) 프론트엔드 배포 (Cloudflare Pages)

- Cloudflare 대시보드 → **Workers & Pages → Create → Pages → 저장소 연결**
- 빌드 설정:
  - **Root directory**: `frontend`
  - **Build command**: `npm ci && npx vite build`
  - **Build output directory**: `dist`
- **환경 변수** (Settings → Environment variables):
  - 빌드 타임(광고, Vite가 인라인): `VITE_ADSENSE_CLIENT`, `VITE_ADSENSE_SLOT`
  - 런타임(프록시 함수용): `BACKEND_URL = https://rhythm-backend.onrender.com`
- `frontend/functions/api/[[path]].ts` 가 자동으로 `/api/*` 요청을 `BACKEND_URL`로
  프록시합니다. 덕분에 프론트는 **같은 도메인**처럼 동작(CORS 불필요)합니다.

> Pages Function 프록시는 큰 파일 업로드/긴 유튜브 다운로드에서 실행 제한에 걸릴
> 수 있습니다. 그럴 땐 선택 A(자체 호스팅)가 더 안정적입니다.

---

## AdSense (Google 광고) 설정

1. [AdSense](https://adsense.google.com) 가입 후 **사이트 추가** → 도메인 등록.
2. **광고 단위** 생성 → `ca-pub-...`(게시자 ID)와 슬롯 ID(10자리)를 받습니다.
3. 값 주입:
   - 자체 호스팅(선택 A): compose에 `VITE_ADSENSE_CLIENT`, `VITE_ADSENSE_SLOT` 전달
   - Cloudflare Pages(선택 B): Pages 환경 변수에 동일하게 설정 후 재배포
4. **`frontend/public/ads.txt`** 의 `pub-0000000000000000` 을 본인 게시자 번호로
   교체하세요(승인에 필요). 배포 후 `https://내도메인/ads.txt` 로 확인됩니다.
5. Google이 사이트를 검토·승인해야 실제 광고가 나옵니다(콘텐츠·트래픽 필요).

- 값이 없으면 광고 코드는 **아예 로드되지 않습니다**(데스크톱 exe/로컬 개발 포함).
- 광고 위치는 랜딩 페이지 하단 1곳입니다. `src/App.tsx` 의 `<AdSlot .../>` 를
  복제해 위치를 늘릴 수 있습니다.

### ⚠️ 정책 경고 — 꼭 읽어주세요
이 앱의 **유튜브 링크 기능(`yt-dlp` 다운로드)** 은 YouTube 약관 위반이며,
AdSense의 "저작권 콘텐츠 다운로드 지원 사이트 금지" 정책에 걸려 **승인 거부 또는
계정 정지** 사유가 될 수 있습니다. 공개·수익화 사이트라면 **사용자가 직접 올린
WAV/MP3 업로드 기능만 노출**하고 유튜브 링크 입력은 숨기거나 제거하길 강력히 권합니다.

---

## (선택) BaaS로 확장하기
로그인, 만든 채보 저장/공유, 갤러리, 조회수 같은 기능은 Supabase/Firebase 같은
BaaS로 붙일 수 있습니다. 이때도 **오디오 분석은 위의 컨테이너 백엔드가 담당**하고,
BaaS는 인증·DB·스토리지만 맡깁니다. 필요하면 이 부분을 이어서 구현해 드릴게요.
```
