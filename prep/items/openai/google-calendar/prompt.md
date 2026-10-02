Design the calendar service behind a personal and team scheduling app. A user creates events on one or more calendars they own, views their schedule by day, week, month or year, and invites other users to an event so that it appears on each invitee's own calendar for them to accept, decline, or mark tentative.

An event has a title, an optional location and description, and a start and end instant; it is created in the creator's local time zone, which is stored with it. An *all-day* event has no time zone at all — it is anchored to calendar dates, not to instants. An event is either a single occurrence or a *recurring event*: a rule that generates occurrences at a frequency — daily; weekly on one or more days of the week (every Tuesday and Thursday); or monthly, either on a fixed day of the month (the 15th) or on the *n*-th of a set of weekdays in the month (the second Tuesday; the last weekday, Monday to Friday) — repeating every *N* periods, and ending after a fixed count, on a given date, or never. Any single occurrence of a recurring event can later be rescheduled or cancelled on its own, identified by the time it was originally scheduled to start, without touching the rest of the series.

The creator of an event can invite other registered users as participants; an invite adds the event to each invitee's own view, and each invitee independently accepts, declines, or marks themselves tentative, without affecting the other invitees' responses.

Scale for this design:

- 100,000,000 registered users, 30,000,000 of whom are active on a given day.
- A user's own calendars hold an average of 120 event definitions with occurrences in any given two-year span — a single occurrence or a whole recurring series each count as one; about 20% of those are recurring series, the rest single occurrences.
- An active user performs an average of 2 write actions a day (create, edit, cancel, or RSVP to an event) and loads 15 calendar views a day (opening the app and switching between day/week/month a few times); usage is 5x busier at its daily peak than the day's average.
- An event averages 2.5 participants, including the organizer; a user is signed in on an average of 2.5 devices.
- Target latency: a view load (day, week, month, or year) under 150 ms at the median and under 400 ms at the 95th percentile; a change made on one device becomes visible on a user's other devices within 5 seconds at the 95th percentile.

In scope: creating, rescheduling and cancelling single and recurring events, including per-occurrence exceptions; time zones, including all-day events; inviting participants and tracking their RSVP; reading the day/week/month/year views; and propagating a change to a user's other devices with low latency, including one that was offline while the change happened. Out of scope: reminder notifications sent ahead of an event's start (push or email); a free/busy lookup across other users' calendars; sharing an entire calendar with another user (as opposed to inviting participants to one event); and importing or exporting events to or from an external calendar system.

Produce:

- Requirements and a scale estimate: total event storage, peak read and write QPS, and the volume of sync traffic pushed to devices.
- A data model (calendars, events, recurrence rules and their exceptions, participants, and whatever a device needs in order to sync) and 3-5 core APIs.
- An architecture diagram, and a walk-through of one event write and one view read along it.
- Deep dives into: how recurring events are stored and turned into concrete occurrences; making the four calendar views fast to read; and keeping a user's devices in sync, including one that was offline.
