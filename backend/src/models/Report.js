import mongoose from "mongoose";

const reportSchema = new mongoose.Schema({
  detected_object: {
    type: String,
    required: true
  },
  axe_x: {
    type: Number,
    required: true  
  },
  axe_y: {
    type: Number,
    required: true
  },
  width: {  
    type: Number,
    required: true
  },  
  height: {
    type: Number,
    required: true
  }
}, { timestamps: true });

export default mongoose.model("Report", reportSchema);
