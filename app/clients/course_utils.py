import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List

from app.utils.path_util import PROJECT_ROOT


COURSES_FILE = PROJECT_ROOT / "output" / "courses.json"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "-", value.strip()).strip("-")
    return slug[:48] or "course"


def _load_courses() -> List[Dict[str, Any]]:
    if not COURSES_FILE.exists():
        return []
    try:
        with open(COURSES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _save_courses(courses: List[Dict[str, Any]]) -> None:
    COURSES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(COURSES_FILE, "w", encoding="utf-8") as f:
        json.dump(courses, f, ensure_ascii=False, indent=2)


def list_courses() -> List[Dict[str, Any]]:
    return sorted(_load_courses(), key=lambda item: item.get("updated_at", ""), reverse=True)


def get_course(course_id: str) -> Dict[str, Any] | None:
    for course in _load_courses():
        if course.get("course_id") == course_id:
            return course
    return None


def ensure_course(course_id: str | None = None, course_name: str | None = None) -> Dict[str, Any]:
    courses = _load_courses()
    name = (course_name or "默认课程").strip()
    if course_id:
        for course in courses:
            if course.get("course_id") == course_id:
                if course_name and course.get("course_name") != course_name:
                    course["course_name"] = course_name
                    course["updated_at"] = _now_iso()
                    _save_courses(courses)
                return course

    for course in courses:
        if (course.get("course_name") or "").strip() == name:
            return course

    base_id = _slugify(name)
    existing_ids = {course.get("course_id") for course in courses}
    new_id = base_id if base_id not in existing_ids else f"{base_id}-{uuid.uuid4().hex[:8]}"
    course = {
        "course_id": new_id,
        "course_name": name,
        "description": "",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    courses.append(course)
    _save_courses(courses)
    return course


def course_response() -> Dict[str, Any]:
    courses = list_courses()
    if not courses:
        courses = [ensure_course(course_name="默认课程")]
    return {"code": 200, "courses": courses}
