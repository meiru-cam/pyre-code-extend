Design the backend for a team chat product used inside companies. Users send direct messages (DMs) to one other person and post in channels shared by a group of people within one company's account, called a *workspace*. A user is commonly signed in from more than one device at once — a desktop app, a phone, a browser tab — and every one of those devices must receive a new message while it is connected. A user must also be notified when new messages arrive and none of their devices is currently connected, and specifically whenever a channel message @-mentions them, subject to a mute or a do-not-disturb schedule. Users attach files to messages and can delete a message they sent, which must remove it for every recipient. Each workspace's channels, messages and files must be completely invisible to every other workspace, even though one backend serves all of them. Most channels a person actually posts into are small, but a handful — a company-wide announcements channel, for instance — can grow to tens of thousands of members.

Scale this design for:

- 200,000 workspaces and 20,000,000 registered users in total, of which 6,000,000 are active on a typical day.
- An active user sends about 35 messages a day on average, across DMs and channels combined — 210,000,000 messages/day system-wide.
- A channel a person is actively posting into averages about 18 members; the largest channels in the largest workspaces run up to roughly 50,000 members.
- At peak, 15% of daily active users are connected at once, each with 1.3 connected devices on average.
- Target delivery latency to an online recipient: 95th percentile under 700 ms from the sender's request to the message appearing on a connected device.

In scope: sending and delivering DM and channel messages to every connected device of every recipient; offline push notifications and @-mention alerts, gated by mute/DND settings; attaching and retrieving files; deleting a message; workspace isolation; and a delivery design that keeps working as a channel grows from a handful of members to tens of thousands. Out of scope: voice/video calls, full-text search over history, editing a sent message, threaded replies as their own object, and the identity provider — assume a directory service can already tell you which users belong to which workspace and channel.

Produce:

- Requirements and a scale estimate: messages sent per second (average and peak); the number of concurrent WebSocket connections and how many gateway instances that requires (state your assumption for how many connections one instance holds); the delivery fan-out volume — deliveries per second to online connections — for an average channel, and separately for one of the largest channels; and the daily and annual growth of message storage.
- A data model (messages, channel membership, per-user read position) and the core APIs: sending a message, paging through a channel's history, deleting a message, marking a channel read, and the WebSocket events a client receives.
- An architecture diagram, and a walk-through of one message from the sender's client to a recipient's connected device.
- Deep dives into message delivery and fan-out; ordering, deduplication and reconnection; and multi-tenancy and sharding.
