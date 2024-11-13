import pythia8
import pandas as pd
import numpy as np
import math

pythia = pythia8.Pythia()

import pythia8

pythia = pythia8.Pythia()

# Enable SoftQCD non-diffractive processes
pythia.readString("SoftQCD:nonDiffractive = on")

# Define the dark photon properties
pythia.readString("9900022:new = gamma_dark DarkPhoton")  # Define new particle
pythia.readString("9900022:spinType = 1")                 # Spin-1 particle (vector boson)
pythia.readString("9900022:chargeType = 0")               # Neutral particle
pythia.readString("9900022:colType = 0")                  # Not colored
pythia.readString("9900022:m0 = 0.002")                   # Set mass to 2 MeV (above 1 MeV)
pythia.readString("9900022:tau0 = 0")                     # Zero lifetime, decays immediately
pythia.readString("9900022:mayDecay = on")                # Allow dark photon to decay
pythia.readString("9900022:isResonance = false")          # Not treated as a resonance
pythia.readString("9900022:addChannel = 1 1.0 0 11 -11")  # Dark photon -> e+ e-

# Add dark photon production channels
pythia.readString("111:addChannel = 1 0.000001 0 22 9900022")  # pi0 -> gamma gamma_dark


# Initialize Pythia
pythia.init()

events_num = 10000
events = []
dark_photon_events = []

for i_event in range(events_num):
    if not pythia.next():
        continue

    particles = []
    sphericity_tensor = np.zeros((3, 3))  # Sphericity tensor initialization

    MET_x = MET_y = 0.0
    HT = visible_energy = visible_px = visible_py = visible_pz = 0.0
    jet_multiplicity = lepton_multiplicity = 0
    jets = []
    leptons = []

    dark_photon_produced = False

    for i in range(pythia.event.size()):
        particle = pythia.event[i]

        if particle.isFinal() and particle.idAbs() not in [12, 14, 16, 9900022]:
            # Store particle properties in a NumPy array for later computations
            particle_data = np.array([particle.px(), particle.py(), particle.pz(), particle.e()])
            particles.append(particle_data)

            HT += particle.pT()
            visible_energy += particle.e()
            visible_px += particle.px()
            visible_py += particle.py()
            visible_pz += particle.pz()

            # Sphericity tensor calculation
            p = particle_data[:3]  # px, py, pz
            sphericity_tensor += np.outer(p, p) / np.dot(p, p)

            # Jets (quarks or gluons)
            if particle.isHadron() and particle.isFinal() and particle.pT() > 0.1:
                jets.append(particle)
                jet_multiplicity += 1

            # Leptons (electron or muon)
            if particle.idAbs() == 11 and particle.isFinal() and particle.pT() > 0.005:  # Electron or positron
                leptons.append(particle)
                lepton_multiplicity += 1


        # Missing Energy (MET)
        if particle.isFinal() and particle.idAbs() in [12, 14, 16, 9900022]:
            MET_x += particle.px()
            MET_y += particle.py()

        # Detect if Dark Photon was produced
        if particle.id() == 9900022:
            print("Dark Photon produced!")
            dark_photon_produced = True

    # MET and delta_phi calculation using NumPy
    MET = np.hypot(MET_x, MET_y)
    delta_phi = np.arctan2(MET_y, MET_x) - np.arctan2(visible_py, visible_px)
    delta_phi = np.arctan2(np.sin(delta_phi), np.cos(delta_phi))

    # Razor variables (using NumPy for faster computation)
    M_T = np.sqrt(2 * HT * MET * (1 - np.cos(delta_phi)))
    MR = np.sqrt(np.maximum(0, visible_energy**2 - visible_pz**2))
    Rsq = (M_T / MR) ** 2 if MR > 0 else 0

    # Invariant Mass of visible particles
    invariant_mass = np.sqrt(np.maximum(0, visible_energy**2 - (visible_px**2 + visible_py**2 + visible_pz**2)))

    # Sum of Transverse Energy
    sum_et = HT

    # Centrality (using NumPy)
    centrality = HT / visible_energy if visible_energy > 0 else 0

    # Eigenvalue-based Sphericity and Aplanarity calculation (NumPy eigenvalue calculation)
    eigenvalues = np.linalg.eigvalsh(sphericity_tensor)
    eigenvalues = np.sort(eigenvalues)[::-1]
    sphericity = 1.5 * (eigenvalues[1] + eigenvalues[2]) if np.sum(eigenvalues) > 0 else 0
    aplanarity = 1.5 * eigenvalues[2] if np.sum(eigenvalues) > 0 else 0

    # Cosine of angle between leading jet and MET
    if jets:
        leading_jet = jets[0]
        cos_theta = (leading_jet.px() * MET_x + leading_jet.py() * MET_y) / (leading_jet.pT() * MET) if MET > 0 else 0
    else:
        cos_theta = 0

    # Delta R (separation between final-state particles) using NumPy broadcasting for pairwise distances
    if len(particles) > 1:
        particles = np.array(particles)
        delta_eta = particles[:, None, 2] - particles[None, :, 2]  # eta differences
        delta_phi_matrix = np.arctan2(np.sin(particles[:, None, 1] - particles[None, :, 1]), 
                                      np.cos(particles[:, None, 1] - particles[None, :, 1]))
        delta_r_matrix = np.sqrt(delta_eta**2 + delta_phi_matrix**2)
        delta_r_avg = np.mean(delta_r_matrix[np.triu_indices(len(particles), k=1)])  # Upper triangular for unique pairs
    else:
        delta_r_avg = 0

    # Store event data
    event_data = {
        "Event Number": i_event,
        "HT": HT,
        "MET": MET,
        "MR": MR,
        "Rsq": Rsq,
        "Invariant Mass": invariant_mass,
        "Sum E_T": sum_et,
        "Centrality": centrality,
        "Sphericity": sphericity,
        "Aplanarity": aplanarity,
        "Cos(Theta) Jet-MET": cos_theta,
        "Delta R (Avg)": delta_r_avg,
        "Jet Multiplicity": jet_multiplicity,
        "Lepton Multiplicity": lepton_multiplicity,
        "Dark Photon Produced": dark_photon_produced,
    }
    events.append(event_data)

    if dark_photon_produced:
        dark_photon_events.append(event_data)

# Print the event number, HT, MET, and confirm dark photon production for stored events
for data_point in dark_photon_events:
    print(data_point)

# Convert to DataFrame
df_events = pd.DataFrame(events)

# Save to CSV file
csv_file_path = "data/sim_with_razor_extended2.csv"
try:
    with open(csv_file_path, "a") as f:
        df_events.to_csv(f, header=f.tell() == 0, index=False)
except FileNotFoundError:
    df_events.to_csv(csv_file_path, index=False)
