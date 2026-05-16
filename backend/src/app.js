import express from "express";
import cors from "cors";
const app = express();

app.use(cors());
app.use(express.json());

import testRoutes from "./routes/test.js";
app.use("/api/test", testRoutes)

import userRoutes from "./routes/userRoutes.js";

app.use("/api/users", userRoutes);


import AIRoutes from "./routes/AI_Routes.js";

app.use("/api/ai", AIRoutes);

import reportRoutes from "./routes/reportsRoutes.js";

app.use("/api/reports", reportRoutes);


export default app;