import re

# -------------------------------
# GPU ISA encoding
# -------------------------------

def encode(op, rd=0, rs1=0, rs2=0, bsel=0, imm=0):
    instr = 0
    instr |= (op & 0xF) << 28
    instr |= (rd & 0x7) << 25
    instr |= (rs1 & 0x7) << 22
    instr |= (rs2 & 0x7) << 19
    instr |= (bsel & 0x3) << 17
    instr |= (imm & 0xFFFF)
    return instr


# -------------------------------
# Opcode Map
# -------------------------------

OP = {
    "NOP": 0x0,
    "LOADI": 0x1,
    "LOAD": 0x2,
    "STORE": 0x3,
    "I16_ADD": 0x4,
    "I16_SUB": 0x5,
    "I16_RELU": 0x7,
    "SET_TID": 0x8,
    "INC_TID": 0x9,
    "LOOP": 0xA,
    "TENSOR_MUL": 0xB,
    "TENSOR_MAC": 0xC,
    "CLEAR_DONE": 0xD,
    "JUMP": 0xE,
    "SET_DONE": 0xF
}


# -------------------------------
# PTX → GPU translation
# -------------------------------

def translate_vector_add(n_words):

    program = []

    # r1 = 0 (SET_TID)
    program.append(encode(OP["SET_TID"], imm=0))

    loop_addr = 1

    # LOAD r1 = A[tid]
    program.append(encode(OP["LOAD"], rd=1, bsel=0, imm=0))

    # LOAD r2 = B[tid]
    program.append(encode(OP["LOAD"], rd=2, bsel=1, imm=0))

    # ADD r3 = r1 + r2
    program.append(encode(OP["I16_ADD"], rd=3, rs1=1, rs2=2))

    # STORE r3 -> C[tid]
    program.append(encode(OP["STORE"], rs2=3, bsel=2, imm=0))

    # INC_TID
    program.append(encode(OP["INC_TID"]))

    # LOOP if tid < n_words
    program.append(encode(OP["LOOP"], imm=loop_addr))

    # DONE
    program.append(encode(OP["SET_DONE"]))

    return program


# -------------------------------
# Output HEX
# -------------------------------

def write_hex(program, filename="vector_add.hex"):
    with open(filename, "w") as f:
        for instr in program:
            f.write(f"{instr:08X}\n")


# -------------------------------
# Example Usage
# -------------------------------

if __name__ == "__main__":

    n = 1024  # vector length

    prog = translate_vector_add(n)

    write_hex(prog)

    print("Generated GPU program:")
    for i, instr in enumerate(prog):
        print(f"{i:02d}: {instr:08X}")