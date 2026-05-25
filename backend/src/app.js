import express from "express";
import cors from "cors";
import userRoutes from "./routes/userRoutes.js";
import AIRoutes from "./routes/AI_Routes.js";
import reportRoutes from "./routes/reportsRoutes.js";

const app = express();

app.use(cors());
app.use(express.json());

if (process.env.NODE_ENV !== "production") {
  const { default: testRoutes } = await import("./routes/test.js");
  app.use("/api/test", testRoutes);
}

app.use("/api/users", userRoutes);
app.use("/api/ai", AIRoutes);
app.use("/api/reports", reportRoutes);

export default app;
