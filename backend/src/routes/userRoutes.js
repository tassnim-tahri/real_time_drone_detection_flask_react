import express from "express";
import {
  getAllUsers,
  updateUserState,
  userDelete,
  login,
  register,
  updateUserPassword,
  getUserByEmail,
  add_user
} from "../controllers/userController.js";

const router = express.Router();

// GET all users
router.get("/", getAllUsers);
router.get("/getUserByEmail/:email", getUserByEmail);

// UPDATE user state (approve)
router.put("/:id/state", updateUserState);
router.put("/:id/password", updateUserPassword);
router.post("/login", login);
router.post("/register", register);
router.post("/add_user", add_user);
router.delete("/:id", userDelete);

export default router;