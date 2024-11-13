import pythia8
import pandas as pd
import numpy as np
import math

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

isolation_cone_radius = 0.4
isolation_threshold = 0.1

for i_event in range(events_num):
    if not pythia.next():
        continue

    particles = []
    sphericity_tensor = np.zeros((3, 3))  # Sphericity tensor initialization

    MET_x = MET_y = 0.0
    HT = visible_energy = visible_px = visible_py = visible_pz = 0.0
    jet_multiplicity = lepton_multiplicity = 0
    jets = []

    dark_photon_produced = False

    for i in range(pythia.event.size()):
        particle = pythia.event[i]

        if particle.isFinal():
            # Missing Energy (MET)
            if particle.idAbs() in [12, 14, 16, 9900022]:
                MET_x += particle.px()
                MET_y += particle.py()
            else:
                # Collect particle properties
                particle_data = [
                    particle.px(), particle.py(), particle.pz(), particle.e(),
                    particle.pT(), particle.eta(), particle.phi(), particle.id()
                ]
                particles.append(particle_data)

                HT += particle.pT()
                visible_energy += particle.e()
                visible_px += particle.px()
                visible_py += particle.py()
                visible_pz += particle.pz()

                # Sphericity tensor calculation
                p_vec = np.array([particle.px(), particle.py(), particle.pz()])
                sphericity_tensor += np.outer(p_vec, p_vec) / np.dot(p_vec, p_vec)

                # Jets (hadrons with pT > 0.1)
                if particle.isHadron() and particle.pT() > 0.1:
                    jets.append(particle)
                    jet_multiplicity += 1

                # Leptons (electrons)
                if particle.idAbs() == 11 and particle.pT() > 0.005:
                    lepton_multiplicity += 1

        # Detect if Dark Photon was produced
        if particle.id() == 9900022:
            print("Dark Photon produced!")
            dark_photon_produced = True

    if len(particles) == 0:
        continue  # Skip event if no particles collected

    particles = np.array(particles)

    # MET and delta_phi calculation
    MET = np.hypot(MET_x, MET_y)
    delta_phi = np.arctan2(MET_y, MET_x) - np.arctan2(visible_py, visible_px)
    delta_phi = np.arctan2(np.sin(delta_phi), np.cos(delta_phi))

    # Razor variables
    M_T = np.sqrt(2 * HT * MET * (1 - np.cos(delta_phi)))
    MR = np.sqrt(np.maximum(0, visible_energy**2 - visible_pz**2))
    Rsq = (M_T / MR) ** 2 if MR > 0 else 0

    # Invariant Mass of visible particles
    invariant_mass = np.sqrt(np.maximum(0, visible_energy**2 - (visible_px**2 + visible_py**2 + visible_pz**2)))

    # Sum of Transverse Energy
    sum_et = HT

    # Centrality
    centrality = HT / visible_energy if visible_energy > 0 else 0

    # Eigenvalue-based Sphericity and Aplanarity calculation
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

    # Delta R (separation between final-state particles)
    if len(particles) > 1:
        delta_eta = particles[:, None, 5] - particles[None, :, 5]  # eta differences
        delta_phi_matrix = np.arctan2(np.sin(particles[:, None, 6] - particles[None, :, 6]),
                                      np.cos(particles[:, None, 6] - particles[None, :, 6]))
        delta_r_matrix = np.sqrt(delta_eta**2 + delta_phi_matrix**2)
        delta_r_avg = np.mean(delta_r_matrix[np.triu_indices(len(particles), k=1)])  # Upper triangular for unique pairs
    else:
        delta_r_avg = 0

    # Isolation calculations
    def calculate_isolation(target_particles, all_particles, cone_radius, isolation_threshold):
        if target_particles.shape[0] == 0:
            return 0, 0.0

        # Compute Delta Eta and Delta Phi differences using broadcasting
        delta_eta = target_particles[:, np.newaxis, 5] - all_particles[np.newaxis, :, 5]
        delta_phi = np.arctan2(
            np.sin(target_particles[:, np.newaxis, 6] - all_particles[np.newaxis, :, 6]),
            np.cos(target_particles[:, np.newaxis, 6] - all_particles[np.newaxis, :, 6])
        )

        # Compute Delta R
        delta_r = np.sqrt(delta_eta**2 + delta_phi**2)

        # Mask to exclude particles within the cone but exclude self-comparison
        isolation_mask = (delta_r < cone_radius) & (delta_r > 0)

        # Calculate sum of pT of particles within the cone for each target particle
        isolation_sum = np.sum(all_particles[:, 4] * isolation_mask, axis=1)

        # Calculate isolation ratio
        isolation_ratio = isolation_sum / target_particles[:, 4]

        # Count isolated particles
        isolated_count = np.sum(isolation_ratio < isolation_threshold)

        # Calculate average isolation ratio
        avg_isolation_ratio = np.mean(isolation_ratio) if target_particles.shape[0] > 0 else 0.0

        return isolated_count, avg_isolation_ratio

    # Extract leptons and photons from particles array
    leptons_array = particles[particles[:, 7] == 11]
    photons_array = particles[particles[:, 7] == 22]

    # Calculate isolation for leptons and photons
    lepton_isolated_count, lepton_avg_isolation = calculate_isolation(
        leptons_array, particles, isolation_cone_radius, isolation_threshold
    )
    photon_isolated_count, photon_avg_isolation = calculate_isolation(
        photons_array, particles, isolation_cone_radius, isolation_threshold
    )

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
        "Lepton Isolated Count": lepton_isolated_count,
        "Lepton Avg Isolation": lepton_avg_isolation,
        "Photon Isolated Count": photon_isolated_count,
        "Photon Avg Isolation": photon_avg_isolation,
        "Dark Photon Produced": dark_photon_produced,
    }
    events.append(event_data)

# Convert to DataFrame
df_events = pd.DataFrame(events)

# Save to CSV file
csv_file_path = "data/sim_with_all_metrics.csv"
df_events.to_csv(csv_file_path, index=False)
