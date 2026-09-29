This is a 55-minute onsite round with one or more interviewers. You choose one past technical project — one you drove yourself, in enough depth to sustain detailed follow-up — and present it for about 20–25 minutes, either from a short slide deck or by sharing your screen to walk through diagrams and code directly; confirm which format your loop expects. A live demonstration of the running system is uncommon. About 25–30 minutes of question-and-answer follow the presentation; a question can also arrive during the presentation itself, before you reach the slide that would have answered it. The questions fall into the following groups.

### Problem and context

- What problem existed before this project, and who had it — a specific team, a specific set of users, a specific metric that was stuck?
- What was in place before your project — nothing, a manual process, an older system — and what was wrong with it?
- Why was this the right project to take on at that time, rather than a year earlier, a year later, or a different problem entirely?
- What was explicitly out of scope, and why?

### Architecture and key decisions

- Walk through the architecture: the main components, and how a request or a piece of data moves through them.
- Which single decision in this design mattered most to how the project turned out?
- Where did you compromise on what you considered the ideal design, and because of what constraint — time, headcount, an existing system you had to integrate with?
- Which part of the design would you defend most strongly if it were pushed on?

### Trade-offs and alternatives rejected

- Why didn't you use a specific, plausible alternative — an existing internal system, a managed product, a simpler design?
- What did you know at the time that ruled that alternative out, as opposed to what you know now?
- Was there a simpler design that would have captured most of the benefit for a fraction of the effort?
- Under what condition would you make the opposite choice today?

### Your contribution versus the team's

- What did you personally design, build, or decide, distinct from what the rest of the team did?
- Who else worked on this project, and what did they own?
- For a part of the system you did not build yourself, what do you actually know about it — its interface, or its internals too?
- Who made the final call on the project's biggest decision — you, or someone else?

### Measurement and results

- What metric did you use to show the project worked, and why that metric rather than another one?
- What was the number before and after, and what was it measured against — a prior baseline, a control group, a rollback?
- Who defined that metric, and how was the target set?
- How do you know the change was caused by your project, and not by something else that changed at the same time?
- Translate the project's cost into a number and its measured return into a number: was it worth it?

### Failures and what you would do differently

- What was the hardest challenge you did not see coming, and how did you find out about it?
- What was the single biggest mistake in the project, and when did you realize it?
- If you started the project again today, what would you do differently?
- What have you carried into a later project because of what this one taught you?

### Scaling and operational concerns

- What happens to the system if load or data volume grows 10x? Which part breaks first?
- What was the actual bottleneck once the system was in production, and how did you find it?
- What is the failure mode when a specific resource is exhausted, and how does the system behave then?
- Who operated the system day to day after launch, and what did that cost?

### For research roles: experimental design

When the project is empirical research — a new method, a measured model behavior, a claim evaluated against baselines — the deep dive adds this group.

- How was the comparison designed — what was held constant, what was varied, and what was the baseline?
- How do you know the result is a real effect and not noise, a leak between training and evaluation data, or an artifact of the evaluation setup?
- What would have shown your hypothesis to be wrong, and did you check for it?
- How many seeds, runs, or repetitions support the headline number, and what was the spread across them?
