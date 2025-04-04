
# Component Responsibilities

This document outlines the responsibilities of each component within the Smart Track Management layered architecture.

## 1. Presentation Layer
- **User Interface (UI)**: Displays real-time track and train information. Receives user inputs and sends them to the business layer.
- **Notification System**: Alerts users of delays, changes, or emergencies.

## 2. Business Logic Layer
- **Train Scheduler**: Handles real-time train scheduling, delay management, and routing.
- **Track Allocator**: Allocates tracks based on availability and train priority.
- **Incident Manager**: Handles operational anomalies and interacts with maintenance systems.

## 3. Data Access Layer
- **Data Repository Accessor**: Interfaces with the database to fetch and update train, track, and schedule information.
- **Log Handler**: Stores system and user logs for future auditing.

## 4. Data Layer
- **Database**: Stores all persistent data including schedules, track statuses, user profiles, and logs.
