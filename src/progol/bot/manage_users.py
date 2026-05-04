"""CLI to manage Progol bot users.

Examples:
    python -m src.progol.bot.manage_users add 123456 --role owner --name "Ivan"
    python -m src.progol.bot.manage_users approve 123456 --role user
    python -m src.progol.bot.manage_users block 123456
    python -m src.progol.bot.manage_users list
"""
import argparse

from src.progol import database


def cmd_add(args):
    database.init_bot_db()
    # In a Telegram private chat, user_id == chat_id. Default to that if the
    # caller didn't pass --user-id explicitly, so the column is populated
    # even on the first seed before the user has DM'd the bot.
    user_id = args.user_id if args.user_id is not None else args.chat_id
    database.bot_upsert_user(
        chat_id=args.chat_id,
        user_id=user_id,
        username=args.username,
        first_name=args.name,
        role=args.role,
        status='active',
    )
    print(f"chat_id={args.chat_id} user_id={user_id} added as {args.role}/active")


def cmd_approve(args):
    database.init_bot_db()
    n = database.bot_set_role(args.chat_id, role=args.role, status='active')
    print(f"chat_id={args.chat_id} -> {args.role}/active ({n} row{'s' if n != 1 else ''})")


def cmd_block(args):
    database.init_bot_db()
    n = database.bot_set_role(args.chat_id, role='blocked', status='blocked')
    print(f"chat_id={args.chat_id} blocked ({n} row{'s' if n != 1 else ''})")


def cmd_list(_args):
    database.init_bot_db()
    users = database.bot_list_users()
    if not users:
        print("(no users)")
        return
    print(f"{'chat_id':>12} {'user_id':>12}  {'role':<8} {'status':<8} {'name':<25} {'username'}")
    for u in users:
        uid = u.get('user_id') if u.get('user_id') is not None else '-'
        print(f"{u['chat_id']:>12} {str(uid):>12}  {u['role']:<8} {u['status']:<8} "
              f"{(u.get('first_name') or ''):<25} {(u.get('username') or '')}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)

    a = sub.add_parser('add')
    a.add_argument('chat_id', type=int)
    a.add_argument('--user-id', type=int, default=None,
                   help='Telegram user_id (defaults to chat_id for private chats)')
    a.add_argument('--role', default='user', choices=['owner', 'admin', 'user'])
    a.add_argument('--name', default=None)
    a.add_argument('--username', default=None)
    a.set_defaults(func=cmd_add)

    a = sub.add_parser('approve')
    a.add_argument('chat_id', type=int)
    a.add_argument('--role', default='user', choices=['owner', 'admin', 'user'])
    a.set_defaults(func=cmd_approve)

    a = sub.add_parser('block')
    a.add_argument('chat_id', type=int)
    a.set_defaults(func=cmd_block)

    a = sub.add_parser('list')
    a.set_defaults(func=cmd_list)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
