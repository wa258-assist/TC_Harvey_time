"""
scripts/notify.py
-----------------
Email notifications for FloodPin pipeline validation events.

Required environment variables:
  SMTP_HOST   - SMTP server hostname (default: smtp.gmail.com)
  SMTP_PORT   - SMTP port (default: 587)
  SMTP_USER   - sender email address
  SMTP_PASS   - sender password / app password

Usage:
  from scripts.notify import notify_validation_errors, notify_pipeline_complete, notify_pipeline_failed
"""
import os, smtplib, traceback
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path

from scripts.utils import get_logger, utcnow_iso

log = get_logger("notify")

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")


def _send(to: str, subject: str, body: str, attachments: list[Path] = None):
    if not (SMTP_USER and SMTP_PASS):
        log.warning("SMTP_USER / SMTP_PASS not set — skipping email to %s", to)
        return False

    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    for path in (attachments or []):
        path = Path(path)
        if not path.exists():
            continue
        part = MIMEBase("application", "octet-stream")
        part.set_payload(path.read_bytes())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{path.name}"')
        msg.attach(part)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.ehlo()
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.sendmail(SMTP_USER, to, msg.as_string())
        log.info("Email sent to %s  subject=%r", to, subject)
        return True
    except Exception:
        log.error("Failed to send email to %s:\n%s", to, traceback.format_exc())
        return False


def notify_validation_errors(to: str, errors: list, event_id: str, date_str: str):
    """Send alert when pin validation finds missing required fields."""
    if not to:
        return
    subject = f"[FloodPin] Validation FAILED — {event_id} {date_str} ({len(errors)} pins)"
    lines = [
        f"FloodPin pipeline validation errors detected.",
        f"",
        f"Event:  {event_id}",
        f"Date:   {date_str}",
        f"Time:   {utcnow_iso()}",
        f"Failed: {len(errors)} pin(s)",
        f"",
        f"Details:",
    ]
    for e in errors[:50]:
        lines.append(f"  pin_id={e.get('pin_id')}  missing={e.get('missing')}")
    if len(errors) > 50:
        lines.append(f"  ... and {len(errors)-50} more")
    _send(to, subject, "\n".join(lines))


def notify_validation_passed(to: str, n_pins: int, event_id: str, date_str: str):
    """Send confirmation when all pins for a day pass validation."""
    if not to:
        return
    subject = f"[FloodPin] Validation OK — {event_id} {date_str} ({n_pins} pins)"
    body = (
        f"All pins passed validation.\n\n"
        f"Event:  {event_id}\n"
        f"Date:   {date_str}\n"
        f"Time:   {utcnow_iso()}\n"
        f"Pins:   {n_pins}\n"
    )
    _send(to, subject, body)


def notify_pipeline_complete(to: str, summary: dict, attachments: list[Path] = None):
    """Send pipeline completion summary with optional CSV attachments."""
    if not to:
        return
    event_id   = summary.get("event_id", "")
    total_pins = summary.get("total_pins", 0)
    mean_conf  = summary.get("mean_confidence", 0)
    runtime_s  = summary.get("runtime_s", 0)
    subject    = f"[FloodPin] Pipeline COMPLETE — {event_id}  {total_pins} pins"

    lines = [
        f"FloodPin pipeline completed successfully.",
        f"",
        f"Event:           {event_id} ({summary.get('event_name','')})",
        f"Total pins:      {total_pins}",
        f"Mean confidence: {mean_conf:.4f}",
        f"Runtime:         {runtime_s:.0f}s",
        f"Generated UTC:   {summary.get('generated_utc','')}",
        f"",
        f"Daily breakdown:",
    ]
    for ds in summary.get("daily_summaries", []):
        lines.append(
            f"  {ds.get('label','')}: {ds.get('n_pins',0)} pins  "
            f"prob={ds.get('mean_hazard_prob',0):.3f}  conf={ds.get('mean_confidence',0):.3f}"
        )
    _send(to, subject, "\n".join(lines), attachments=attachments)


def notify_pipeline_failed(to: str, reason: str, event_id: str):
    """Send alert when the pipeline aborts early."""
    if not to:
        return
    subject = f"[FloodPin] Pipeline FAILED — {event_id}"
    body = (
        f"FloodPin pipeline failed.\n\n"
        f"Event:  {event_id}\n"
        f"Time:   {utcnow_iso()}\n"
        f"Reason: {reason}\n"
    )
    _send(to, subject, body)
