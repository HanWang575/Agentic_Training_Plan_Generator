"""Create missing completed-week Program Generator summaries for all local users.

Run this script each Sunday with Windows Task Scheduler or another local scheduler.
"""

from sqlmodel import Session, select

from app.core.config import get_settings
from app.program.service import ProgramService
from app.storage.database import engine, init_db
from app.storage.models import UserAccount


def main() -> None:
    init_db()
    with Session(engine) as session:
        service = ProgramService(session=session, settings=get_settings())
        users = session.exec(select(UserAccount)).all()
        for user in users:
            result = service.refresh_weekly_summaries(user.user_id)
            print(f"{user.user_id}: {len(result.notes)} completed weekly summaries available")


if __name__ == "__main__":
    main()
