"""Speech Studio 的界面后端：一套 RPC 分发 + 状态推送 + 媒体解析。

搬进来的 React 界面（web/）只认两套后端——桌面宿主桥与浏览器预览。这里提供第三套：
浏览器直接访问 FastAPI，RPC 走 `POST /api/yovoice/call`，状态经 `WS /ws/yovoice` 推送。
方法名与 `web/src/shared/lib/preview-backend.ts` 一一对应；未实现的返回
`@yovoice.error.methodUnsupported`，界面只显示一句错误提示，不会静默失败。
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

from . import yovoice_bridge as bridge
from .config import Config
from .models import PodcastProject
from .podcast_manager import PodcastManager
from .sample_manager import SampleManager

logger = logging.getLogger("tts_service.yovoice")

ERROR_UNSUPPORTED = "@yovoice.error.methodUnsupported"
ERROR_REQUEST_INVALID = "@yovoice.error.requestInvalid"
ERROR_BUSY = "@yovoice.error.busy"
ERROR_AUDIO_MISSING = "@yovoice.error.audioMissing"
ERROR_AUDIO_KIND = "@yovoice.error.audioKindInvalid"
ERROR_AUDIO_DURATION = "@yovoice.error.audioDuration"
ERROR_NAME_LENGTH = "@yovoice.error.nameLength"

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

# yovoice 服务模型目录的缓存时长与请求超时（秒）。
_REMOTE_MODELS_TTL = 5.0
_REMOTE_MODELS_TIMEOUT = 3.0

_duration_cache: dict[tuple[str, float], float] = {}


def audio_duration(path: Path) -> float:
    try:
        stat = path.stat()
    except OSError:
        return 0.0
    key = (str(path), stat.st_mtime)
    cached = _duration_cache.get(key)
    if cached is not None:
        return cached
    duration = 0.0
    try:
        import soundfile as sf

        duration = float(sf.info(str(path)).duration)
    except Exception:
        logger.debug("无法读取音频时长：%s", path, exc_info=True)
    _duration_cache[key] = duration
    return duration


class RpcError(Exception):
    """带消息码的调用失败，界面按码翻译成提示文案。"""

    def __init__(self, code: str, params: Optional[dict[str, Any]] = None):
        super().__init__(code)
        self.code = code
        self.params = params or {}


class YovoiceHub:
    """把状态推给所有打开的界面；工作线程改完状态后调用 notify()。"""

    def __init__(self, build_state: Callable[[], dict[str, Any]]):
        self._build_state = build_state
        self._clients: set[WebSocket] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self) -> None:
        self._loop = asyncio.get_running_loop()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)
        try:
            await websocket.send_json({"event": "state", "state": self._build_state()})
            while True:
                # 界面不会主动发言，这里只是等断开。
                await websocket.receive_text()
        except Exception:
            pass
        finally:
            self._clients.discard(websocket)

    def notify(self) -> None:
        if not self._clients or self._loop is None or self._loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(), self._loop)
        except RuntimeError:
            logger.debug("状态推送跳过：事件循环已关闭")

    async def _broadcast(self) -> None:
        if not self._clients:
            return
        payload = {"event": "state", "state": self._build_state()}
        for websocket in list(self._clients):
            try:
                await websocket.send_json(payload)
            except Exception:
                self._clients.discard(websocket)


class YovoiceBackend:
    """状态、偏好与生成任务的持有者；也是 RPC 分发的落点。"""

    def __init__(
        self,
        *,
        config: Config,
        sample_manager: SampleManager,
        podcast_manager: PodcastManager,
        outputs_dir: Path,
        engine_provider: Callable[[], Any],
        preferences_path: Path,
        live_speakers: Iterable[str] = (),
    ):
        self.config = config
        self.sample_manager = sample_manager
        self.podcast_manager = podcast_manager
        self.outputs_dir = Path(outputs_dir)
        self.engine_provider = engine_provider
        self.preferences_path = Path(preferences_path)
        self.live_speakers = list(live_speakers)
        self._preferences = self._load_preferences()
        self._characters: list[dict[str, Any]] = []
        self._ui_state_path = self.preferences_path.with_name("yovoice-ui.json")
        self._load_ui_state()
        self._activity: Optional[dict[str, Any]] = None
        self._cancel = threading.Event()
        self._lock = threading.RLock()
        # yovoice 服务里的模型目录：TTL 缓存，避免每次 state.get 都打网络。
        self._remote_models_cache: Optional[tuple[float, list[dict[str, Any]], list[dict[str, Any]]]] = None
        self._remote_models_lock = threading.Lock()
        self._local_api = {
            "enabled": False,
            "port": 8080,
            "token": "speech-studio".ljust(64, "0"),
            "running": False,
        }
        self.hub = YovoiceHub(self.state)

    # ── 状态 ────────────────────────────────────────────────

    def _load_preferences(self) -> dict[str, Any]:
        defaults = {
            "downloadSource": "modelscope",
            "backend": "metal" if _is_apple_silicon() else "cpu",
            "modelDirectory": None,
            "uiLocale": "zh-CN",
            "introSeen": True,
            "tourSeen": True,
        }
        try:
            stored = json.loads(self.preferences_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stored = {}
        if isinstance(stored, dict):
            defaults.update({key: value for key, value in stored.items() if value is not None})
        return defaults

    def _load_ui_state(self) -> None:
        try:
            stored = json.loads(self._ui_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(stored, dict) and isinstance(stored.get("characters"), list):
            self._characters = stored["characters"]

    def _save_preferences(self) -> None:
        try:
            self.preferences_path.write_text(
                json.dumps(self._preferences, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            logger.warning("偏好写入失败：%s", self.preferences_path, exc_info=True)

    def _save_ui_state(self) -> None:
        try:
            self._ui_state_path.write_text(
                json.dumps({"characters": self._characters}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            logger.warning("界面状态写入失败：%s", self._ui_state_path, exc_info=True)

    def voices(self) -> list[dict[str, Any]]:
        voices: list[dict[str, Any]] = []
        for sample in self.sample_manager.list_samples():
            voices.append({
                "id": sample.speaker,
                "name": sample.display_name,
                "fileName": f"{sample.file_stem}.wav",
                "duration": audio_duration(sample.wav_path),
                "referenceText": sample.transcript,
                "source": "import",
            })
        return voices

    def state(self) -> dict[str, Any]:
        with self._lock:
            activity = dict(self._activity) if self._activity else None
            characters = [dict(item) for item in self._characters]
        catalog, installed = self._remote_models()
        state = bridge.build_state(
            projects=self.podcast_manager.list_projects(),
            voices=self.voices(),
            preferences=dict(self._preferences),
            live_speakers=self.live_speakers,
            activity=activity,
            runtime_backend=self._preferences.get("backend"),
            runtime_version=None,
            models=installed,
            runtime_path=self._models_endpoint()[0] or None,
        )
        state["characters"] = characters
        return state

    def catalog(self) -> list[dict[str, Any]]:
        """界面「模型」页展示用的完整目录（含未安装项）。"""
        return self._remote_models()[0]

    def _models_endpoint(self) -> tuple[str, str]:
        base = getattr(self.config.model, "yovoice_base_url", "") or ""
        token = getattr(self.config.model, "yovoice_api_token", "") or ""
        return base.rstrip("/"), token

    def _default_model(self) -> dict[str, Any]:
        """yovoice 不可达时的兜底模型，取配置里的 model id，保证生成不被门禁卡住。"""
        model_id = getattr(self.config.model, "yovoice_model_id", "") or "index-2.5-q8"
        return {"id": model_id, "path": "", "managed": True}

    def _fetch_remote_models(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        base, token = self._models_endpoint()
        if not base:
            return [], []
        request = urllib.request.Request(f"{base}/v1/models")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(request, timeout=_REMOTE_MODELS_TIMEOUT) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.warning("无法读取 yovoice 模型目录（%s），改用默认模型：%s", base, exc)
            return [], []
        if not isinstance(payload, dict):
            return [], []
        catalog = payload.get("catalog")
        installed = payload.get("installed")
        return (
            catalog if isinstance(catalog, list) else [],
            installed if isinstance(installed, list) else [],
        )

    def _remote_models(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """返回 (目录, 已安装)；带 TTL 缓存，yovoice 不可达时回退到配置的默认模型。"""
        now = time.monotonic()
        with self._remote_models_lock:
            cached = self._remote_models_cache
            if cached and now - cached[0] < _REMOTE_MODELS_TTL:
                return list(cached[1]), list(cached[2])
        catalog, installed = self._fetch_remote_models()
        if not installed:
            installed = [self._default_model()]
        if not catalog:
            catalog = [{**self._default_model(), "name": installed[0]["id"]}]
        with self._remote_models_lock:
            self._remote_models_cache = (now, catalog, installed)
        return list(catalog), list(installed)

    def notify(self) -> None:
        self.hub.notify()

    # ── 生成任务 ────────────────────────────────────────────

    def _set_activity(self, status: str, **fields: Any) -> None:
        with self._lock:
            self._activity = {"status": status, **fields}

    def _run_generation(self, project_id: str, pending: list[int], total: int) -> None:
        started = bridge.now_iso()
        engine = None
        for done, index in enumerate(pending, start=1):
            if self._cancel.is_set():
                self._set_activity(
                    "cancelled", kind="generate", code=bridge.ACTIVITY_CANCELLED,
                    projectId=project_id, total=total, received=done - 1, startedAt=started,
                    modelId=None, cueId=None, characterId=None, requestId=None,
                    params=None, errorCode=None, errorParams=None,
                )
                self.notify()
                return
            try:
                if engine is None:
                    engine = self.engine_provider()
                self.podcast_manager.regenerate_segment(
                    project_id=project_id, index=index, engine=engine, sample_manager=self.sample_manager
                )
            except Exception as exc:  # noqa: BLE001 - 失败要如实回报给界面
                logger.exception("播客分段生成失败：%s #%s", project_id, index)
                self._set_activity(
                    "failed", kind="generate", code=bridge.ACTIVITY_FAILED,
                    projectId=project_id, total=total, received=done - 1, startedAt=started,
                    modelId=None, cueId=None, characterId=None, requestId=None,
                    params=None, errorCode=bridge.ERROR_GENERATE_FAILED,
                    errorParams={"detail": str(exc)},
                )
                self.notify()
                return
            self._set_activity(
                "running", kind="generate", code=bridge.ACTIVITY_GENERATE,
                projectId=project_id, total=total, received=done, startedAt=started,
                modelId=None, cueId=None, characterId=None, requestId=None,
                params=None, errorCode=None, errorParams=None,
            )
            self.notify()
        self._set_activity(
            "completed", kind="generate", code=bridge.ACTIVITY_COMPLETED,
            projectId=project_id, total=total, received=total, startedAt=started,
            modelId=None, cueId=None, characterId=None, requestId=None,
            params=None, errorCode=None, errorParams=None,
        )
        self.notify()

    # ── RPC 分发 ────────────────────────────────────────────

    def handle(self, method: str, data: Any) -> Any:
        handler = getattr(self, f"_rpc_{method.replace('.', '_')}", None)
        if handler is None:
            raise RpcError(ERROR_UNSUPPORTED)
        return handler(data if data is not None else {})

    # 状态与偏好
    def _rpc_state_get(self, _data: Any) -> dict[str, Any]:
        return {
            "state": self.state(),
            "catalog": self.catalog(),
            "engine": {"version": "", "minimum": ""},
        }

    def _rpc_preferences_save(self, data: Any) -> bool:
        if isinstance(data, dict):
            self._preferences = {**self._preferences, **data}
            self._save_preferences()
        return True

    def _rpc_sidebar_save(self, _data: Any) -> bool:
        return True

    def _rpc_inspector_available(self, _data: Any) -> bool:
        return True

    def _rpc_assistant_entry(self, _data: Any) -> bool:
        return True

    def _rpc_diag_client(self, data: Any) -> bool:
        detail = data.get("detail") if isinstance(data, dict) else None
        logger.info("界面事件：%s %s", data.get("event") if isinstance(data, dict) else "", detail or "")
        return True

    # 作品
    def _rpc_draft_save(self, data: Any) -> bool:
        project_id = str((data or {}).get("id") or "").strip()
        if not project_id:
            raise RpcError(ERROR_REQUEST_INVALID)
        existing = self.podcast_manager.get_project(project_id)
        fields = bridge.draft_to_project(data, existing=existing)
        self.podcast_manager.replace_segments(
            project_id, title=fields["title"], segments=fields["segments"], created_at=fields["createdAt"]
        )
        self.notify()
        return True

    def _rpc_draft_delete(self, data: Any) -> bool:
        project_id = str((data or {}).get("id") or "")
        with self._lock:
            running = bool(self._activity and self._activity.get("status") == "running")
            busy_project = (self._activity or {}).get("projectId")
        if running and busy_project == project_id:
            raise RpcError(ERROR_BUSY)
        self.podcast_manager.delete_project(project_id)
        self.notify()
        return True

    # 生成
    def _rpc_generation_start(self, data: Any) -> bool:
        project_id = str((data or {}).get("id") or "").strip()
        if not project_id:
            raise RpcError(ERROR_REQUEST_INVALID)
        with self._lock:
            if self._activity and self._activity.get("status") == "running":
                raise RpcError(ERROR_BUSY)
        existing = self.podcast_manager.get_project(project_id)
        fields = bridge.draft_to_project(data, existing=existing)
        project: PodcastProject = self.podcast_manager.replace_segments(
            project_id, title=fields["title"], segments=fields["segments"], created_at=fields["createdAt"]
        )
        pending = [index for index, segment in enumerate(project.segments) if segment.status == "pending"]
        self._cancel.clear()
        if not pending:
            self._set_activity(
                "completed", kind="generate", code=bridge.ACTIVITY_COMPLETED,
                projectId=project_id, total=0, received=0, startedAt=bridge.now_iso(),
                modelId=None, cueId=None, characterId=None, requestId=None,
                params=None, errorCode=None, errorParams=None,
            )
            self.notify()
            return True
        self._set_activity(
            "running", kind="generate", code=bridge.ACTIVITY_GENERATE,
            projectId=project_id, total=len(pending), received=0, startedAt=bridge.now_iso(),
            modelId=None, cueId=None, characterId=None, requestId=None,
            params=None, errorCode=None, errorParams=None,
        )
        self.notify()
        threading.Thread(
            target=self._run_generation, args=(project_id, pending, len(pending)), daemon=True
        ).start()
        return True

    def _rpc_operation_cancel(self, _data: Any) -> bool:
        with self._lock:
            activity = self._activity
        if activity and activity.get("status") == "running":
            self._cancel.set()
        return True

    def _rpc_generation_lyricTimes(self, _data: Any) -> bool:
        return True

    # 音色
    def _rpc_voice_record(self, data: Any) -> dict[str, Any]:
        name = str((data or {}).get("name") or "").strip()
        payload = (data or {}).get("base64") or ""
        if not name or not payload:
            raise RpcError(ERROR_REQUEST_INVALID)
        audio = base64.b64decode(payload)
        if len(audio) > MAX_UPLOAD_BYTES:
            raise RpcError("@yovoice.error.audioTooLarge")
        try:
            sample = self.sample_manager.add_voice(name, audio, "", overwrite=True)
        except ValueError as exc:
            raise RpcError(ERROR_AUDIO_DURATION, {"detail": str(exc)}) from exc
        self.notify()
        return {
            "id": sample.speaker,
            "name": sample.display_name,
            "fileName": f"{sample.file_stem}.wav",
            "duration": audio_duration(sample.wav_path),
            "referenceText": sample.transcript,
            "source": "import",
        }

    def _rpc_voice_update(self, data: Any) -> dict[str, Any]:
        voice_id = str((data or {}).get("id") or "")
        sample = self.sample_manager.get(voice_id)
        if sample is None:
            raise RpcError(ERROR_AUDIO_MISSING)
        name = str((data or {}).get("name") or "").strip()
        if name and name != sample.speaker:
            if len(name) > 100:
                raise RpcError(ERROR_NAME_LENGTH)
            sample = self._rename_voice(sample, name)
        reference_text = (data or {}).get("referenceText")
        if reference_text is not None:
            sample = self.sample_manager.update_transcript(sample.speaker, str(reference_text))
        self.notify()
        return {
            "id": sample.speaker,
            "name": sample.display_name,
            "fileName": f"{sample.file_stem}.wav",
            "duration": audio_duration(sample.wav_path),
            "referenceText": sample.transcript,
            "source": "import",
        }

    def _rename_voice(self, sample, name: str):
        stem = name[:-len(".normal")] if name.lower().endswith(".normal") else name
        target_wav = sample.wav_path.with_name(f"{stem}.wav")
        if target_wav.exists():
            raise RpcError("@yovoice.error.nameTaken", {"name": stem})
        sample.wav_path.rename(target_wav)
        sample.txt_path.rename(sample.txt_path.with_name(f"{stem}.txt"))
        sample.cache_path.unlink(missing_ok=True)
        self.sample_manager.refresh()
        renamed = self.sample_manager.resolve(stem)
        if renamed is None:
            raise RpcError(ERROR_AUDIO_MISSING)
        return renamed

    # 资源
    def _rpc_assets_list(self, _data: Any) -> list[dict[str, Any]]:
        return bridge.build_assets(
            projects=self.podcast_manager.list_projects(),
            voices=self.voices(),
            voice_durations={},
        )

    def _rpc_assets_delete(self, data: Any) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for item in (data or {}).get("items") or []:
            kind = item.get("kind")
            try:
                if kind not in ("generation", "segment", "voice"):
                    raise RpcError(ERROR_AUDIO_KIND)
                self._delete_media("voices" if kind == "voice" else "outputs", str(item.get("id") or ""))
                results.append({"kind": kind, "id": item.get("id")})
            except RpcError as exc:
                results.append({"kind": kind, "id": item.get("id"), "error": exc.code, "params": exc.params})
        return results

    def _rpc_assets_image(self, _data: Any) -> None:
        raise RpcError(ERROR_AUDIO_MISSING)

    def _delete_media(self, kind: str, item_id: str) -> None:
        if kind == "voices":
            if self.sample_manager.get(item_id) is None:
                raise RpcError(ERROR_AUDIO_MISSING)
            self.sample_manager.delete_voice(item_id)
            return
        parsed = bridge.parse_segment_token(item_id)
        if parsed is None:
            raise RpcError(ERROR_AUDIO_KIND)
        project_id, filename = parsed
        path = self.podcast_manager.project_audio_dir(project_id) / filename
        path.unlink(missing_ok=True)
        project = self.podcast_manager.get_project(project_id)
        if project is not None:
            for segment in project.segments:
                if segment.audio_filename == filename:
                    segment.audio_filename = None
                    segment.status = "pending"
            self.podcast_manager.save_project(project)

    def _rpc_media_delete(self, data: Any) -> bool:
        self._delete_media(str((data or {}).get("kind") or ""), str((data or {}).get("id") or ""))
        self.notify()
        return True

    def _rpc_media_rename(self, data: Any) -> bool:
        name = str((data or {}).get("name") or "").strip()
        if not name or len(name) > 120:
            raise RpcError(ERROR_NAME_LENGTH)
        if (data or {}).get("kind") == "voices":
            sample = self.sample_manager.get(str(data.get("id") or ""))
            if sample is None:
                raise RpcError(ERROR_AUDIO_MISSING)
            self._rename_voice(sample, name)
            self.notify()
        return True

    def _rpc_media_reveal(self, _data: Any) -> bool:
        return True

    def _rpc_timeline_import(self, data: Any) -> dict[str, Any]:
        payload = (data or {}).get("base64") or ""
        name = str((data or {}).get("name") or "导入音频").strip()[:120]
        if not payload:
            raise RpcError(ERROR_REQUEST_INVALID)
        audio = base64.b64decode(payload)
        if len(audio) > MAX_UPLOAD_BYTES:
            raise RpcError("@yovoice.error.audioTooLarge")
        asset_id = uuid.uuid4().hex
        filename = f"import-{asset_id}.wav"
        target = self.outputs_dir / filename
        target.write_bytes(audio)
        duration = audio_duration(target)
        if duration < 0.01 or duration > 3600:
            target.unlink(missing_ok=True)
            raise RpcError("@yovoice.timeline.importDuration")
        return {"id": asset_id, "name": name, "fileName": filename, "duration": duration}

    # 角色：与音色库一样只保留名字与示例文本，不做试听推理。
    def _rpc_character_save(self, data: Any) -> dict[str, Any]:
        character = dict(data or {})
        if not str(character.get("name") or "").strip():
            raise RpcError("@yovoice.error.characterInvalid")
        character["id"] = character.get("id") or uuid.uuid4().hex
        character["updatedAt"] = bridge.now_iso()
        self._characters = [item for item in self._characters if item.get("id") != character["id"]]
        self._characters.append(character)
        self._save_ui_state()
        self.notify()
        return character

    def _rpc_character_delete(self, data: Any) -> bool:
        character_id = str((data or {}).get("id") or "")
        self._characters = [item for item in self._characters if item.get("id") != character_id]
        self._save_ui_state()
        self.notify()
        return True

    # 日志
    def _rpc_logs_query(self, _data: Any) -> dict[str, Any]:
        return {"entries": [], "total": 0, "modules": []}

    def _rpc_logs_export(self, _data: Any) -> bool:
        return True

    def _rpc_logs_open(self, _data: Any) -> bool:
        return True

    # 存储与本机接口
    def _rpc_storage_info(self, _data: Any) -> dict[str, Any]:
        outputs = _directory_bytes(self.outputs_dir)
        voices = _directory_bytes(self.sample_manager.base_dir)
        return {
            "root": str(self.outputs_dir.parent),
            "modelsBytes": 0,
            "outputsBytes": outputs,
            "voicesBytes": voices,
            "downloadsBytes": 0,
            "totalBytes": outputs + voices,
            "pendingPath": None,
        }

    def _rpc_storage_directory(self, _data: Any) -> bool:
        return True

    def _rpc_storage_directory_open(self, _data: Any) -> bool:
        return True

    def _rpc_localApi_get(self, _data: Any) -> dict[str, Any]:
        return dict(self._local_api)

    def _rpc_localApi_save(self, data: Any) -> dict[str, Any]:
        port = (data or {}).get("port")
        if not isinstance(port, int) or not 1024 <= port <= 65535:
            raise RpcError("@yovoice.error.localApiPort")
        self._local_api["enabled"] = bool((data or {}).get("enabled"))
        self._local_api["port"] = port
        self._local_api["running"] = self._local_api["enabled"]
        return dict(self._local_api)

    def _rpc_localApi_resetToken(self, _data: Any) -> dict[str, Any]:
        self._local_api["token"] = uuid.uuid4().hex + uuid.uuid4().hex
        return dict(self._local_api)

    # 不支持的调用：明确回报，不静默成功
    def _rpc_ai_codex_status(self, _data: Any) -> dict[str, Any]:
        return {}

    def _unsupported(self, data: Any = None) -> Any:  # noqa: ARG002 - 统一签名
        raise RpcError(ERROR_UNSUPPORTED)


def _is_apple_silicon() -> bool:
    import platform

    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _directory_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


# ── 路由注册 ────────────────────────────────────────────────


def register_yovoice_api(app: FastAPI, backend: YovoiceBackend) -> None:
    outputs_dir = backend.outputs_dir.resolve()
    voices_dir = backend.sample_manager.base_dir.resolve()

    @app.post("/api/yovoice/call", include_in_schema=False)
    def yovoice_call(payload: dict[str, Any]) -> dict[str, Any]:
        method = str(payload.get("method") or "")
        try:
            return {"result": backend.handle(method, payload.get("data"))}
        except RpcError as exc:
            logger.info("界面调用被拒绝：%s -> %s", method, exc.code)
            return {"error": {"code": exc.code, "params": exc.params}}
        except Exception as exc:  # noqa: BLE001 - 兜底，避免 500 让界面只看到“未知错误”
            logger.exception("界面调用失败：%s", method)
            return {"error": {"code": "@yovoice.error.unknown", "params": {"detail": str(exc)}}}

    @app.websocket("/ws/yovoice")
    async def yovoice_socket(websocket: WebSocket) -> None:
        backend.hub.bind_loop()
        await backend.hub.connect(websocket)

    @app.get("/api/yovoice/media/{kind}/{name:path}", include_in_schema=False)
    def yovoice_media(kind: str, name: str) -> FileResponse:
        target = _resolve_media(kind, name, outputs_dir, voices_dir)
        if target is None:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Media not found")
        return FileResponse(target, media_type="audio/wav")


def _resolve_media(kind: str, name: str, outputs_dir: Path, voices_dir: Path) -> Optional[Path]:
    if kind == "voices":
        target = (voices_dir / name).resolve()
        if target.parent != voices_dir or not target.is_file():
            return None
        return target
    if kind != "outputs":
        return None
    parsed = bridge.parse_segment_token(name)
    if parsed is not None:
        target = (outputs_dir / "podcasts" / parsed[0] / parsed[1]).resolve()
    else:
        target = (outputs_dir / name).resolve()
        if target.parent != outputs_dir:
            return None
    return target if target.is_file() else None
