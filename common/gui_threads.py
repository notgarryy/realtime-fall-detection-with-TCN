# General Library Imports
import numpy as np
import time
import os
# PyQt imports
from PySide2.QtCore import QThread, Signal, QMutexLocker, QMutex, QWaitCondition
import pyqtgraph as pg

# Local Imports
from gui_parser import UARTParser
from gui_common import *
from graph_utilities import *

import joblib
from tcn import TCN
from tensorflow.keras.saving import load_model

# Logger
import logging
log = logging.getLogger(__name__)

from collections import deque

TIMESTEPS = 30
MODEL_FILE = "./Model/TCN_Model.h5"
SCALER_FILE = "./Model/TCN_Scaler.pkl"

# Classifier Configurables
MAX_NUM_TRACKS = 20

# Expected minimums and maximums to bound the range of colors used for coloring points
SNR_EXPECTED_MIN = 5
SNR_EXPECTED_MAX = 40
SNR_EXPECTED_RANGE = SNR_EXPECTED_MAX - SNR_EXPECTED_MIN
DOPPLER_EXPECTED_MIN = -30
DOPPLER_EXPECTED_MAX = 30
DOPPLER_EXPECTED_RANGE = DOPPLER_EXPECTED_MAX - DOPPLER_EXPECTED_MIN

# Different methods to color the points
COLOR_MODE_SNR = 'SNR'
COLOR_MODE_HEIGHT = 'Height'
COLOR_MODE_DOPPLER = 'Doppler'
COLOR_MODE_TRACK = 'Associated Track'

# Magic Numbers for Target Index TLV
TRACK_INDEX_WEAK_SNR = 253
TRACK_INDEX_BOUNDS = 254
TRACK_INDEX_NOISE = 255


class parseUartThread(QThread): 
    fin = Signal(dict) 
    batchReady = Signal(np.ndarray) 
    
    def __init__(self, uParser, timesteps=TIMESTEPS): 
        super().__init__() 
        self.parser = uParser 
        self.timesteps = timesteps 
        self.timestamp = time.strftime("%m%d%Y%H%M%S") 
        self.outputDir = f'./dataset/{self.timestamp}' 
        os.makedirs(self.outputDir, exist_ok=True)
        
        self.buffer = deque(maxlen=timesteps)
        self.running = True 
        
    def run(self): 
        parse_func = ( 
            self.parser.readAndParseUartSingleCOMPort 
            if self.parser.parserType == "SingleCOMPort" 
            else self.parser.readAndParseUartDoubleCOMPort
        ) 

        while self.running:
            outputDict = parse_func() 
            self.fin.emit(outputDict) 
            
            frameJSON = {
                'frameData': outputDict,
                'timestamp': time.time() * 1000 
            } 
            
            batch = self.processData(frameJSON)
            
            if batch is not None: 
                self.batchReady.emit(batch)

    def processData(self, frameJSON):
        frameData = frameJSON.get("frameData", {})
        trackData = frameData.get("trackData")
        
        if trackData is None or len(trackData) == 0:
            return None
        
        trackData = np.array(trackData)
        mask = (trackData[:,1] != 0) & (trackData[:,2] != 0) & (trackData[:,3] != 0)
        rows = trackData[mask, 1:10]
        
        if rows.size == 0:
            return None
        
        for row in rows:
            self.buffer.append(row)
        
        if len(self.buffer) < self.timesteps:
            return None
        
        window = np.array(list(self.buffer)[-self.timesteps:])
        
        zero_count = np.sum(np.all(window == 0, axis=1))
        if zero_count > 0:
            logging.warning(f"Buffer still has {zero_count} zero frames!")
            return None  
        
        return window.reshape(1, self.timesteps, window.shape[1])
        
    def stop(self): 
        self.running = False 
        self.wait()

class preprocessThread(QThread):
    preprocessedReady = Signal(np.ndarray, float)

    def __init__(self, scaler_path=f"{SCALER_FILE}", num_features=9):
        super().__init__()
        self.scaler = joblib.load(scaler_path)
        print(f"Scaler loaded.")
        print(f"Scaler path used: {scaler_path}")
        self.queue = deque(maxlen=5)
        self.num_features = num_features
        self.running = True

        self.mutex = QMutex()
        self.new_data_event = QWaitCondition()

        self.auto_align_xyz = True
        self.align_history = deque(maxlen=50)

    def addBatch(self, batch):
        """Add a new data batch for preprocessing."""
        batch = np.asarray(batch)
        if batch.ndim == 2:
            batch = np.expand_dims(batch, axis=0)

        with QMutexLocker(self.mutex):
            self.queue.append(batch)
            self.new_data_event.wakeOne()  

    def run(self):
        """Thread loop for preprocessing radar point clouds."""
        self.avg_prob_prev = 0.5
        while self.running:
            self.mutex.lock()
            if not self.queue:
                self.new_data_event.wait(self.mutex)
            if not self.running:
                self.mutex.unlock()
                break
            if self.scaler is None:
                continue

            batch = self.queue.popleft()
            self.mutex.unlock()

            if batch is None or batch.size == 0:
                continue

            flat = batch.reshape(-1, batch.shape[-1])
            if flat.size == 0:
                continue

            azimuth_deg = (np.degrees(np.arctan2(flat[:, 1], flat[:, 0])) + 357.0) % 360.0
            avg_azimuth = np.mean(azimuth_deg)

            angle_diff = ((azimuth_deg - 90.0 + 180.0) % 360.0) - 180.0
            abs_diff = np.abs(angle_diff)
            lift_strength = np.cos(np.radians(abs_diff))
            lift_strength = np.clip(lift_strength, 0.0, 1.0)
            z_lift = 0.6 * lift_strength
            mask_far = (azimuth_deg < 50.0) | (azimuth_deg > 130.0)
            z_lift[mask_far] *= 0.6
            az_center_offset = np.abs(azimuth_deg - 90.0)
            z_scale = 1.0 - np.clip(az_center_offset / 90.0, 0.0, 0.55)

            flat_comp = flat.copy()
            flat_comp[:, 2] = flat[:, 2] * z_scale + 0.5 * z_lift

            pos_scale, vel_scale, acc_scale = 0.8, 1.1, 1.2
            flat_comp[:, 0:3] *= pos_scale
            flat_comp[:, 3:6] *= vel_scale
            flat_comp[:, 6:9] *= acc_scale

            if np.ptp(azimuth_deg) > 30:
                flat_comp[:, 2] = 0.8 * flat[:, 2] + 0.2 * flat_comp[:, 2]

            try:
                reshaped = flat_comp.reshape(-1, self.num_features)

                if self.auto_align_xyz:
                    live_mean_xyz = reshaped[:, :3].mean(axis=0)
                    train_mean_xyz = self.scaler.mean_[:3]
                    offset = train_mean_xyz - live_mean_xyz
                    reshaped[:, :3] += offset 
                    self.align_history.append(offset)
                    if len(self.align_history) % 10 == 0:  
                        avg_offset = np.mean(np.vstack(self.align_history), axis=0)
                        logging.debug(f"[AlignXYZ] Avg offset (x,y,z): {avg_offset.round(3)}")

                logging.debug(f"[Scaling] Input mean/std before: {reshaped.mean(axis=0)}, {reshaped.std(axis=0)}")
                scaled = self.scaler.transform(reshaped).reshape(batch.shape)
                logging.debug(f"[Scaling] Output mean/std after: {scaled.mean():.3f}, {scaled.std():.3f}")

                self.preprocessedReady.emit(scaled.copy(), avg_azimuth)

            except Exception as e:
                logging.exception(f"[preprocessThread] Scaling error: {e}")
                continue

    def stop(self):
        with QMutexLocker(self.mutex):
            self.running = False
        self.new_data_event.wakeOne()
        self.wait(500)

class predictThread(QThread):
    predictionReady = Signal(str)

    def __init__(
        self,
        model_path=f"{MODEL_FILE}",
        vote_window=10,              
        use_impact_check=False,      
        accel_threshold=1.5,         
        parent=None,
    ):
        super().__init__(parent)
        try:
            self.model = load_model(model_path, custom_objects={'TCN': TCN}, compile=False)
        except Exception as e:
            logging.error(f"[predictThread] Failed to load model: {e}")
            self.model = None

        self.queue = deque(maxlen=6)
        self.running = True

        self.vote_window = vote_window
        self.fall_buffer = deque(maxlen=vote_window)
        self.others_buffer = deque(maxlen=vote_window)

        self.avg_prob_prev = 0.5
        self.last_emit_time = 0.0
        self.fall_counter = 0.0
        self.recover_counter = 0.0
        self.fall_state = "Others"
        self.mutex = QMutex()

        self.fall_confirm_frames = 3      # frames of consistent evidence to confirm FALL
        self.recover_confirm_frames = 8   # frames to go back to Others
        self.alpha_secondary = 0.12       # secondary exponential smoothing (small)
        self.min_ratio_for_fall = 1.8     # avg_fall / avg_others must exceed this
        self.min_avg_fall_prob = 0.55     # avg fall probability threshold
        self.use_impact_check = use_impact_check
        self.accel_threshold = accel_threshold

    def addPreprocessed(self, batch, avg_azimuth=90.0):
        batch = np.asarray(batch)
        if batch.ndim == 2:
            batch = np.expand_dims(batch, axis=0)
        with QMutexLocker(self.mutex):
            self.queue.append((batch, avg_azimuth))

    def run(self):
        if self.model is None:
            logging.error("[predictThread] No model loaded, exiting thread.")
            return

        while self.running:
            with QMutexLocker(self.mutex):
                if not self.queue:
                    pass

            if not self.queue:
                self.msleep(10)
                continue

            with QMutexLocker(self.mutex):
                batch, avg_azimuth = self.queue.popleft()

            if batch is None or batch.size == 0:
                continue

            try:
                pred = self.model.predict(batch, verbose=0)[0]   
                prob_others, prob_fall = float(pred[0]), float(pred[1])

                self.fall_buffer.append(prob_fall)
                self.others_buffer.append(prob_others)

                avg_prob_fall = float(np.mean(self.fall_buffer))
                avg_prob_others = float(np.mean(self.others_buffer))

                eps = 1e-6
                ratio = (avg_prob_fall + eps) / (avg_prob_others + eps)

                avg_prob_fall = self.alpha_secondary * avg_prob_fall + (1 - self.alpha_secondary) * self.avg_prob_prev
                self.avg_prob_prev = avg_prob_fall

                impact_ok = True
                if self.use_impact_check:
                    try:
                        arr = np.asarray(batch)
                        if arr.ndim >= 3 and arr.shape[-1] >= 9:
                            # acceleration features assumed in cols 6:9 (ax,ay,az)
                            acc = arr[0, :, 6:9]
                            acc_mag = np.linalg.norm(acc, axis=1)
                            peak = float(np.max(np.abs(acc_mag)))
                            impact_ok = peak >= self.accel_threshold
                        else:
                            impact_ok = False
                    except Exception:
                        impact_ok = False

                az_offset = abs(avg_azimuth - 90.0)
                az_factor = np.clip(az_offset / 90.0, 0.0, 0.25)
                min_avg_fall = self.min_avg_fall_prob - az_factor * 0.05
                min_ratio = self.min_ratio_for_fall - az_factor * 0.3

                if (avg_prob_fall >= min_avg_fall and ratio >= min_ratio and impact_ok):
                    self.fall_counter = min(self.fall_counter + 1.0, 100.0)
                    self.recover_counter = max(self.recover_counter - 0.6, 0.0)
                elif avg_prob_others >= 0.6:
                    self.recover_counter = min(self.recover_counter + 1.0, 100.0)
                    self.fall_counter = max(self.fall_counter - 0.6, 0.0)
                else:
                    self.fall_counter = max(self.fall_counter - 0.3, 0.0)
                    self.recover_counter = max(self.recover_counter - 0.3, 0.0)

                if self.fall_counter >= self.fall_confirm_frames and self.fall_state != "FALL!":
                    self.fall_state = "FALL!"
                    logging.info("[predictThread] FALL confirmed!")
                    self.recover_counter = 0.0

                elif self.recover_counter >= self.recover_confirm_frames and self.fall_state != "Others":
                    self.fall_state = "Others"
                    logging.info("[predictThread] Recovered to normal.")
                    self.fall_counter = 0.0

                if self.fall_state == "FALL!":
                    confidence = avg_prob_fall * 100.0
                else:
                    confidence = avg_prob_others * 100.0

                result_str = (
                    f"{self.fall_state} {confidence:.1f}%"
                )

                now = time.time()
                if now - self.last_emit_time > 0.12:  
                    self.predictionReady.emit(result_str)
                    self.last_emit_time = now

                logging.debug(
                    f"[predictThread] pred=[{prob_others:.3f}, {prob_fall:.3f}] avgF={avg_prob_fall:.3f} avgO={avg_prob_others:.3f} "
                    f"ratio={ratio:.2f} fall_cnt={self.fall_counter:.1f} rec_cnt={self.recover_counter:.1f} impact_ok={impact_ok}"
                )

            except Exception as e:
                logging.exception(f"[predictThread] Error during prediction: {e}")
                continue

    def stop(self):
        with QMutexLocker(self.mutex):
            self.running = False
        self.wait(500)

class sendCommandThread(QThread):
    done = Signal()

    def __init__(self, uParser, command):
        QThread.__init__(self)
        self.parser = uParser
        self.command = command

    def run(self):
        self.parser.sendLine(self.command)
        self.done.emit()

class updateQTTargetThread3D(QThread):
    done = Signal()

    def __init__(self, pointCloud, targets, scatter, pcplot, numTargets, ellipsoids, coords, colorGradient=None, classifierOut=[], zRange=[-3, 3], pointColorMode="", drawTracks=True, trackColorMap=None, pointBounds={'enabled': False}):
        QThread.__init__(self)
        self.pointCloud = pointCloud
        self.targets = targets
        self.scatter = scatter
        self.pcplot = pcplot
        self.colorArray = ('r', 'g', 'b', 'w')
        self.numTargets = numTargets
        self.ellipsoids = ellipsoids
        self.coordStr = coords
        self.classifierOut = classifierOut
        self.zRange = zRange
        self.colorGradient = colorGradient
        self.pointColorMode = pointColorMode
        self.drawTracks = drawTracks
        self.trackColorMap = trackColorMap
        self.pointBounds = pointBounds
        np.seterr(divide='ignore')

    def drawTrack(self, track, trackColor):
        tid = int(track[0])
        x = track[1]
        y = track[2]
        z = track[3]

        track = self.ellipsoids[tid]
        mesh = getBoxLinesCoords(x, y, z)
        track.setData(pos=mesh, color=trackColor, width=2,
                      antialias=True, mode='lines')
        track.setVisible(True)

    def getPointColors(self, i):
        if (self.pointBounds['enabled']):
            xyz_coords = self.pointCloud[i, 0:3]
            if (xyz_coords[0] < self.pointBounds['minX']
                        or xyz_coords[0] > self.pointBounds['maxX']
                        or xyz_coords[1] < self.pointBounds['minY']
                        or xyz_coords[1] > self.pointBounds['maxY']
                        or xyz_coords[2] < self.pointBounds['minZ']
                        or xyz_coords[2] > self.pointBounds['maxZ']
                    ) :
                return pg.glColor((0, 0, 0, 0))

        if (self.pointColorMode == COLOR_MODE_SNR):
            snr = self.pointCloud[i, 4]
            if (snr < SNR_EXPECTED_MIN) or (snr > SNR_EXPECTED_MAX):
                return pg.glColor('w')
            else:
                return pg.glColor(self.colorGradient.getColor((snr-SNR_EXPECTED_MIN)/SNR_EXPECTED_RANGE))

        elif (self.pointColorMode == COLOR_MODE_HEIGHT):
            zs = self.pointCloud[i, 2]

            if (zs < self.zRange[0]) or (zs > self.zRange[1]):
                return pg.glColor('w')
            else:
                colorRange = self.zRange[1]+abs(self.zRange[0])
                zs = self.zRange[1] - zs
                return pg.glColor(self.colorGradient.getColor(abs(zs/colorRange)))

        elif (self.pointColorMode == COLOR_MODE_DOPPLER):
            doppler = self.pointCloud[i, 3]
            if (doppler < DOPPLER_EXPECTED_MIN) or (doppler > DOPPLER_EXPECTED_MAX):
                return pg.glColor('w')
            else:
                return pg.glColor(self.colorGradient.getColor((doppler-DOPPLER_EXPECTED_MIN)/DOPPLER_EXPECTED_RANGE))

        elif (self.pointColorMode == COLOR_MODE_TRACK):
            trackIndex = int(self.pointCloud[i, 6])
            if (trackIndex == TRACK_INDEX_WEAK_SNR or trackIndex == TRACK_INDEX_BOUNDS or trackIndex == TRACK_INDEX_NOISE):
                return pg.glColor('w')
            else:
                try:
                    return self.trackColorMap[trackIndex]
                except Exception as e:
                    log.error(e)
                    return pg.glColor('w')

        else:
            return pg.glColor('g')

    def run(self):

        # if self.pointCloud is None or len(self.pointCloud) == 0:
        #     print("Point Cloud is empty or None.")
        # else:
        #     print("Point Cloud Shape:", self.pointCloud.shape)

        # Clear all previous targets
        for e in self.ellipsoids:
            if (e.visible()):
                e.hide()
        try:
            # Create a list of just X, Y, Z values to be plotted
            if (self.pointCloud is not None):
                toPlot = self.pointCloud[:, 0:3]
                # print("Data for Visualization:", toPlot)

                # Determine the size of each point based on its SNR
                with np.errstate(divide='ignore'):
                    size = np.log2(self.pointCloud[:, 4])

                # Each color is an array of 4 values, so we need an numPoints*4 size 2d array to hold these values
                pointColors = np.zeros((self.pointCloud.shape[0], 4))

                # Set the color of each point
                for i in range(self.pointCloud.shape[0]):
                    pointColors[i] = self.getPointColors(i)

                # Plot the points
                self.scatter.setData(pos=toPlot, color=pointColors, size=size)
                # Debugging
                # print("Pos Data for Visualization:", toPlot)
                # print("Color Data for Visualization:", pointColors)
                # print("Size Data for Visualization:", size)

                # Make the points visible
                self.scatter.setVisible(True)
            else:
                # Make the points invisible if none are detected.
                self.scatter.setVisible(False)
        except Exception as e:
            log.error(
                "Unable to draw point cloud, ignoring and continuing execution...")
            print("Unable to draw point cloud, ignoring and continuing execution...")
            print(f"Error in point cloud visualization: {e}")

        # Graph the targets
        try:
            if (self.drawTracks):
                if (self.targets is not None):
                    for track in self.targets:
                        trackID = int(track[0])
                        trackColor = self.trackColorMap[trackID]
                        self.drawTrack(track, trackColor)
        except:
            log.error(
                "Unable to draw all tracks, ignoring and continuing execution...")
            print("Unable to draw point cloud, ignoring and continuing execution...")
            print(f"Error in point cloud visualization: {e}")
        self.done.emit()

    def stop(self):
        self.terminate()
