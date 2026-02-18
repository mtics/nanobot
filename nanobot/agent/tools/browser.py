"""Optional browser automation tools (Playwright). Requires pip install nanobot-ai[browser] and playwright install chromium."""

from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False
    Browser = None  # type: ignore
    BrowserContext = None  # type: ignore
    Page = None  # type: ignore
    Playwright = None  # type: ignore

BROWSER_NOT_INSTALLED_MSG = (
    "Error: Browser tools require playwright. Install with: pip install nanobot-ai[browser] and run: playwright install chromium"
)


class BrowserController:
    """Shared browser instance for browser_* tools. Lifecycle managed by AgentLoop."""

    def __init__(
        self,
        workspace: Path,
        headless: bool = True,
        timeout_ms: int = 30000,
        proxy_server: str = "",
        storage_state_path: str = "",
    ):
        self.workspace = workspace
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.proxy_server = proxy_server or None
        self._storage_path = storage_state_path.strip() or str(workspace / "browser" / "cookie.json")
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    async def launch(self) -> None:
        if not HAS_PLAYWRIGHT:
            return
        self._playwright = await async_playwright().start()
        opts: dict[str, Any] = {"headless": self.headless}
        if self.proxy_server:
            opts["proxy"] = {"server": self.proxy_server}
        self._browser = await self._playwright.chromium.launch(**opts)
        ctx_opts: dict[str, Any] = {}
        path = Path(self._storage_path)
        if path.exists():
            ctx_opts["storage_state"] = self._storage_path
        self._context = await self._browser.new_context(**ctx_opts)
        self._context.set_default_timeout(self.timeout_ms)
        self._page = await self._context.new_page()

    async def close(self) -> None:
        if self._context:
            try:
                await self._context.close()
            except Exception:
                pass
            self._context = None
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None
        self._page = None

    async def _ensure_page(self) -> str | None:
        if not HAS_PLAYWRIGHT:
            return BROWSER_NOT_INSTALLED_MSG
        if not self._page:
            await self.launch()
        if not self._page:
            return "Error: Browser failed to start."
        return None

    async def navigate(self, url: str) -> str:
        err = await self._ensure_page()
        if err:
            return err
        try:
            await self._page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            path = Path(self._storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            await self._context.storage_state(path=self._storage_path)
            return f"Navigated to {url}"
        except Exception as e:
            return f"Error navigating to {url}: {e}"

    async def snapshot(self) -> str:
        err = await self._ensure_page()
        if err:
            return err
        try:
            content = await self._page.content()
            return content[:50000] if len(content) > 50000 else content
        except Exception as e:
            return f"Error getting snapshot: {e}"

    async def click(self, selector: str) -> str:
        err = await self._ensure_page()
        if err:
            return err
        try:
            await self._page.click(selector, timeout=self.timeout_ms)
            return f"Clicked: {selector}"
        except Exception as e:
            return f"Error clicking {selector}: {e}"

    async def type_text(self, selector: str, text: str) -> str:
        err = await self._ensure_page()
        if err:
            return err
        try:
            await self._page.fill(selector, text)
            return f"Typed into {selector}"
        except Exception as e:
            return f"Error typing into {selector}: {e}"

    async def press_key(self, key: str) -> str:
        err = await self._ensure_page()
        if err:
            return err
        try:
            await self._page.keyboard.press(key)
            return f"Pressed key: {key}"
        except Exception as e:
            return f"Error pressing key {key}: {e}"

    async def save_session(self) -> str:
        err = await self._ensure_page()
        if err:
            return err
        try:
            path = Path(self._storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            await self._context.storage_state(path=self._storage_path)
            return f"Saved session to {self._storage_path}"
        except Exception as e:
            return f"Error saving session: {e}"


def _make_browser_tools(controller: BrowserController) -> list[Tool]:
    """Build the six browser tools that use the shared controller. Returns [] if playwright not installed."""

    if not HAS_PLAYWRIGHT:
        return []

    class BrowserNavigateTool(Tool):
        name = "browser_navigate"
        description = "Navigate the browser to a URL. Session (cookies/storage) is persisted after navigation."
        parameters = {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Full URL to open"}},
            "required": ["url"],
        }

        async def execute(self, url: str, **kwargs: Any) -> str:
            return await controller.navigate(url)

    class BrowserSnapshotTool(Tool):
        name = "browser_snapshot"
        description = "Get the current page HTML content (for parsing or inspection)."
        parameters = {"type": "object", "properties": {}, "required": []}

        async def execute(self, **kwargs: Any) -> str:
            return await controller.snapshot()

    class BrowserClickTool(Tool):
        name = "browser_click"
        description = "Click an element on the page using a CSS selector."
        parameters = {
            "type": "object",
            "properties": {"selector": {"type": "string", "description": "CSS selector of the element to click"}},
            "required": ["selector"],
        }

        async def execute(self, selector: str, **kwargs: Any) -> str:
            return await controller.click(selector)

    class BrowserTypeTool(Tool):
        name = "browser_type"
        description = "Type text into an input element identified by CSS selector."
        parameters = {
            "type": "object",
            "properties": {
                "selector": {"type": "string", "description": "CSS selector of the input"},
                "text": {"type": "string", "description": "Text to type"},
            },
            "required": ["selector", "text"],
        }

        async def execute(self, selector: str, text: str, **kwargs: Any) -> str:
            return await controller.type_text(selector, text)

    class BrowserPressTool(Tool):
        name = "browser_press"
        description = "Press a keyboard key (e.g. Enter, Tab, Escape)."
        parameters = {
            "type": "object",
            "properties": {"key": {"type": "string", "description": "Key to press (e.g. Enter, Tab)"}},
            "required": ["key"],
        }

        async def execute(self, key: str, **kwargs: Any) -> str:
            return await controller.press_key(key)

    class BrowserSaveSessionTool(Tool):
        name = "browser_save_session"
        description = "Explicitly save browser session (cookies, localStorage) to disk for persistence."
        parameters = {"type": "object", "properties": {}, "required": []}

        async def execute(self, **kwargs: Any) -> str:
            return await controller.save_session()

    return [
        BrowserNavigateTool(),
        BrowserSnapshotTool(),
        BrowserClickTool(),
        BrowserTypeTool(),
        BrowserPressTool(),
        BrowserSaveSessionTool(),
    ]
