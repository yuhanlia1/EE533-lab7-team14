`timescale 1ns/1ps

`define UDP_REG_ADDR_WIDTH 23
`define CPCI_NF2_DATA_WIDTH 32

module gpu_regfile_64 #(parameter NREG = 32) (
  input              clk,
  input              rst,
  input      [4:0]   ra0,
  output reg [63:0]  rd0,
  input      [4:0]   ra1,
  output reg [63:0]  rd1,
  input      [4:0]   ra2,
  output reg [63:0]  rd2,
  input              we,
  input      [4:0]   wa,
  input      [63:0]  wd
);
  reg [63:0] rf [0:NREG-1];
  integer i;
  always @(posedge clk) begin
    if (rst) begin
      for (i = 0; i < NREG; i = i + 1) rf[i] <= 64'd0;
    end else begin
      if (we && (wa != 5'd0)) rf[wa] <= wd;
    end
  end
  always @(*) begin
    rd0 = (ra0 == 5'd0) ? 64'd0 : rf[ra0];
    rd1 = (ra1 == 5'd0) ? 64'd0 : rf[ra1];
    rd2 = (ra2 == 5'd0) ? 64'd0 : rf[ra2];
  end
endmodule

module gpu_exec_i16 (
  input  [1:0]  op,
  input  [63:0] a,
  input  [63:0] b,
  output [63:0] y
);
  wire signed [15:0] a0;
  wire signed [15:0] a1;
  wire signed [15:0] a2;
  wire signed [15:0] a3;
  wire signed [15:0] b0;
  wire signed [15:0] b1;
  wire signed [15:0] b2;
  wire signed [15:0] b3;

  assign a0 = a[15:0];
  assign a1 = a[31:16];
  assign a2 = a[47:32];
  assign a3 = a[63:48];

  assign b0 = b[15:0];
  assign b1 = b[31:16];
  assign b2 = b[47:32];
  assign b3 = b[63:48];

  reg [15:0] y0;
  reg [15:0] y1;
  reg [15:0] y2;
  reg [15:0] y3;

  function [15:0] relu16;
    input signed [15:0] x;
    begin
      if (x[15]) relu16 = 16'h0000;
      else relu16 = x;
    end
  endfunction

  always @(*) begin
    y0 = 16'h0; y1 = 16'h0; y2 = 16'h0; y3 = 16'h0;
    case (op)
      2'd0: begin y0 = a0 + b0; y1 = a1 + b1; y2 = a2 + b2; y3 = a3 + b3; end
      2'd1: begin y0 = a0 - b0; y1 = a1 - b1; y2 = a2 - b2; y3 = a3 - b3; end
      2'd2: begin y0 = relu16(a0); y1 = relu16(a1); y2 = relu16(a2); y3 = relu16(a3); end
      default: begin y0 = 16'h0; y1 = 16'h0; y2 = 16'h0; y3 = 16'h0; end
    endcase
  end

  assign y = {y3, y2, y1, y0};
endmodule

module gpu_tensor_core_pipe3 (
  input         clk,
  input         rst,
  input         in_valid,
  input  [1:0]  in_op,
  input  [63:0] in_a,
  input  [63:0] in_b,
  input  [63:0] in_c,
  output        out_valid,
  output [63:0] out_y
);
  function [15:0] bf16_pack;
    input s;
    input [7:0] exp;
    input [6:0] frac;
    begin
      bf16_pack = {s, exp, frac};
    end
  endfunction

  function [15:0] bf16_relu;
    input [15:0] x;
    begin
      if (x[15]) bf16_relu = 16'h0000;
      else bf16_relu = x;
    end
  endfunction

  function [15:0] bf16_mul;
    input [15:0] a;
    input [15:0] b;
    reg sa;
    reg sb;
    reg sr;
    reg [7:0] ea;
    reg [7:0] eb;
    reg [7:0] er;
    reg [6:0] fa;
    reg [6:0] fb;
    reg [7:0] sig_a;
    reg [7:0] sig_b;
    reg [15:0] prod;
    reg [7:0] sig8;
    reg [6:0] frac;
    reg guard;
    reg sticky;
    reg roundb;
    begin
      sa = a[15]; sb = b[15]; ea = a[14:7]; eb = b[14:7];
      fa = a[6:0]; fb = b[6:0];
      if ((ea == 8'd0 && fa == 7'd0) || (eb == 8'd0 && fb == 7'd0)) begin
        bf16_mul = 16'h0000;
      end else begin
        sr = sa ^ sb;
        sig_a = {1'b1, fa};
        sig_b = {1'b1, fb};
        prod = sig_a * sig_b;
        er = ea + eb - 8'd127;
        if (prod[15]) begin
          sig8 = prod[15:8];
          guard = prod[7];
          sticky = |prod[6:0];
          er = er + 8'd1;
        end else begin
          sig8 = prod[14:7];
          guard = prod[6];
          sticky = |prod[5:0];
        end
        frac = sig8[6:0];
        roundb = guard & (sticky | frac[0]);
        if (roundb) begin
          if (frac == 7'h7F) begin
            frac = 7'h00;
            er = er + 8'd1;
          end else begin
            frac = frac + 7'd1;
          end
        end
        bf16_mul = bf16_pack(sr, er, frac);
      end
    end
  endfunction

  function [15:0] bf16_add;
    input [15:0] a;
    input [15:0] b;
    reg sa;
    reg sb;
    reg sr;
    reg [7:0] ea;
    reg [7:0] eb;
    reg [7:0] er;
    reg [6:0] fa;
    reg [6:0] fb;
    reg [7:0] sig_a;
    reg [7:0] sig_b;
    reg [10:0] ext_a;
    reg [10:0] ext_b;
    reg [10:0] a_al;
    reg [10:0] b_al;
    integer sh;
    integer i;
    reg sticky;
    reg [11:0] sum;
    reg [11:0] mag;
    reg [8:0] sig9;
    reg [6:0] frac;
    reg guard;
    reg roundb;
    reg stickyb;
    reg inc;
    begin
      sa = a[15]; sb = b[15]; ea = a[14:7]; eb = b[14:7];
      fa = a[6:0]; fb = b[6:0];
      if (ea == 8'd0 && fa == 7'd0) begin
        bf16_add = b;
      end else if (eb == 8'd0 && fb == 7'd0) begin
        bf16_add = a;
      end else begin
        sig_a = {1'b1, fa};
        sig_b = {1'b1, fb};
        ext_a = {sig_a, 3'b000};
        ext_b = {sig_b, 3'b000};
        er = (ea >= eb) ? ea : eb;
        a_al = ext_a;
        b_al = ext_b;

        if (er > ea) begin
          sh = er - ea;
          if (sh >= 11) begin
            sticky = |a_al;
            a_al = 11'd0;
            if (sticky) a_al[0] = 1'b1;
          end else if (sh != 0) begin
            sticky = 1'b0;
            for (i = 0; i < 11; i = i + 1) begin
              if (i < sh) sticky = sticky | a_al[i];
            end
            a_al = a_al >> sh;
            if (sticky) a_al[0] = 1'b1;
          end
        end

        if (er > eb) begin
          sh = er - eb;
          if (sh >= 11) begin
            sticky = |b_al;
            b_al = 11'd0;
            if (sticky) b_al[0] = 1'b1;
          end else if (sh != 0) begin
            sticky = 1'b0;
            for (i = 0; i < 11; i = i + 1) begin
              if (i < sh) sticky = sticky | b_al[i];
            end
            b_al = b_al >> sh;
            if (sticky) b_al[0] = 1'b1;
          end
        end

        if (sa == sb) begin
          sum = {1'b0, a_al} + {1'b0, b_al};
          sr = sa;
          mag = sum;
          if (mag[11]) begin
            sticky = mag[0];
            mag = mag >> 1;
            if (sticky) mag[0] = 1'b1;
            er = er + 8'd1;
          end
        end else begin
          if (a_al >= b_al) begin
            mag = {1'b0, a_al} - {1'b0, b_al};
            sr = sa;
          end else begin
            mag = {1'b0, b_al} - {1'b0, a_al};
            sr = sb;
          end
        end

        if (mag == 12'd0) begin
          bf16_add = 16'h0000;
        end else begin
          for (i = 0; i < 12; i = i + 1) begin
            if (mag[10] == 1'b0 && er != 8'd0) begin
              mag = mag << 1;
              er = er - 8'd1;
            end
          end

          guard = mag[2];
          roundb = mag[1];
          stickyb = mag[0];
          frac = mag[9:3];
          inc = guard & (roundb | stickyb | frac[0]);
          sig9 = {1'b0, mag[10:3]} + {8'd0, inc};
          if (sig9[8]) begin
            er = er + 8'd1;
            sig9 = sig9 >> 1;
          end
          frac = sig9[6:0];
          bf16_add = bf16_pack(sr, er, frac);
        end
      end
    end
  endfunction

  reg        v0;
  reg [1:0]  op0;
  reg [63:0] a0;
  reg [63:0] b0;
  reg [63:0] c0;

  reg        v1;
  reg [1:0]  op1;
  reg [63:0] a1;
  reg [63:0] b1;
  reg [63:0] c1;

  reg        v2;
  reg [63:0] y2;

  wire [15:0] a0_0;
  wire [15:0] a0_1;
  wire [15:0] a0_2;
  wire [15:0] a0_3;
  wire [15:0] b0_0;
  wire [15:0] b0_1;
  wire [15:0] b0_2;
  wire [15:0] b0_3;

  assign a0_0 = a0[15:0];
  assign a0_1 = a0[31:16];
  assign a0_2 = a0[47:32];
  assign a0_3 = a0[63:48];
  assign b0_0 = b0[15:0];
  assign b0_1 = b0[31:16];
  assign b0_2 = b0[47:32];
  assign b0_3 = b0[63:48];

  reg [15:0] p1_0;
  reg [15:0] p1_1;
  reg [15:0] p1_2;
  reg [15:0] p1_3;

  wire [15:0] sum01;
  wire [15:0] sum23;
  wire [15:0] sum0123;
  wire [15:0] sum_relu;
  assign sum01 = bf16_add(p1_0, p1_1);
  assign sum23 = bf16_add(p1_2, p1_3);
  assign sum0123 = bf16_add(sum01, sum23);
  assign sum_relu = bf16_relu(sum0123);

  always @(posedge clk) begin
    if (rst) begin
      v0 <= 1'b0;
      op0 <= 2'd0;
      a0 <= 64'd0;
      b0 <= 64'd0;
      c0 <= 64'd0;
    end else begin
      v0 <= in_valid;
      op0 <= in_op;
      a0 <= in_a;
      b0 <= in_b;
      c0 <= in_c;
    end
  end

  always @(posedge clk) begin
    if (rst) begin
      v1 <= 1'b0;
      op1 <= 2'd0;
      a1 <= 64'd0;
      b1 <= 64'd0;
      c1 <= 64'd0;
      p1_0 <= 16'd0;
      p1_1 <= 16'd0;
      p1_2 <= 16'd0;
      p1_3 <= 16'd0;
    end else begin
      v1 <= v0;
      op1 <= op0;
      a1 <= a0;
      b1 <= b0;
      c1 <= c0;
      if (op0 == 2'd0 || op0 == 2'd1 || op0 == 2'd3) begin
        p1_0 <= bf16_mul(a0_0, b0_0);
        p1_1 <= bf16_mul(a0_1, b0_1);
        p1_2 <= bf16_mul(a0_2, b0_2);
        p1_3 <= bf16_mul(a0_3, b0_3);
      end else begin
        p1_0 <= 16'd0;
        p1_1 <= 16'd0;
        p1_2 <= 16'd0;
        p1_3 <= 16'd0;
      end
    end
  end

  wire [15:0] c1_0;
  wire [15:0] c1_1;
  wire [15:0] c1_2;
  wire [15:0] c1_3;
  assign c1_0 = c1[15:0];
  assign c1_1 = c1[31:16];
  assign c1_2 = c1[47:32];
  assign c1_3 = c1[63:48];

  always @(posedge clk) begin
    if (rst) begin
      v2 <= 1'b0;
      y2 <= 64'd0;
    end else begin
      v2 <= v1;
      if (!v1) begin
        y2 <= 64'd0;
      end else begin
        case (op1)
          2'd0: y2 <= {p1_3, p1_2, p1_1, p1_0};
          2'd1: y2 <= {bf16_add(p1_3, c1_3), bf16_add(p1_2, c1_2), bf16_add(p1_1, c1_1), bf16_add(p1_0, c1_0)};
          2'd2: y2 <= {bf16_relu(a1[63:48]), bf16_relu(a1[47:32]), bf16_relu(a1[31:16]), bf16_relu(a1[15:0])};
          2'd3: y2 <= {48'h0, sum_relu};
          default: y2 <= 64'd0;
        endcase
      end
    end
  end

  assign out_valid = v2;
  assign out_y = y2;
endmodule

module gpu_tensor_unit (
  input             clk,
  input             rst,
  input             in_valid,
  input      [1:0]  in_op,
  input      [63:0] in_a,
  input      [63:0] in_b,
  input      [63:0] in_c,
  output            out_valid,
  output     [63:0] out_y
);
  gpu_tensor_core_pipe3 u (
    .clk(clk),
    .rst(rst),
    .in_valid(in_valid),
    .in_op(in_op),
    .in_a(in_a),
    .in_b(in_b),
    .in_c(in_c),
    .out_valid(out_valid),
    .out_y(out_y)
  );
endmodule

module gpu_imem #(parameter DEPTH = 256, parameter AW = 8) (
  input               clk,
  input  [AW-1:0]      raddr,
  output reg [31:0]    rdata,
  input               we,
  input  [AW-1:0]      waddr,
  input  [31:0]        wdata
);
  reg [31:0] mem [0:DEPTH-1];
  integer i;
  initial begin
    for (i = 0; i < DEPTH; i = i + 1) mem[i] = 32'h0;
    rdata = 32'h0;
  end
  always @(posedge clk) begin
    if (we) mem[waddr] <= wdata;
    rdata <= mem[raddr];
  end
endmodule

module gpu_dmem #(parameter DEPTH = 1024, parameter AW = 10) (
  input               clk,
  input               we,
  input  [AW-1:0]      waddr,
  input  [63:0]        wdata,
  input  [AW-1:0]      raddr,
  output reg [63:0]    rdata,
  input  [AW-1:0]      dbg_raddr,
  output reg [63:0]    dbg_rdata
);
  reg [63:0] mem [0:DEPTH-1];
  integer j;
  initial begin
    for (j = 0; j < DEPTH; j = j + 1) mem[j] = 64'h0;
    rdata = 64'h0;
    dbg_rdata = 64'h0;
  end
  always @(posedge clk) begin
    if (we) mem[waddr] <= wdata;
    rdata <= mem[raddr];
    dbg_rdata <= mem[dbg_raddr];
  end
endmodule

module gpu_core #(
  parameter IMEM_DEPTH = 256,
  parameter DMEM_DEPTH = 1024,
  parameter IMEM_AW    = 8,
  parameter DMEM_AW    = 10
)(
  input                clk,
  input                rst,
  input                run,
  input  [9:0]         n_words,
  input  [DMEM_AW-1:0] a_base,
  input  [DMEM_AW-1:0] b_base,
  input  [DMEM_AW-1:0] c_base,
  output reg           done,

  output [IMEM_AW-1:0] dbg_pc,
  output [IMEM_AW-1:0] dbg_pcf,
  output [31:0]        dbg_ir,
  output [9:0]         dbg_tid,
  output               dbg_ld_wait,
  output               dbg_pending,
  output [DMEM_AW-1:0] dbg_eff_addr,
  output               dbg_rf_we,
  output [4:0]         dbg_rf_rd,
  output [63:0]        dbg_rf_wdata,
  output               dbg_st_we,
  output [DMEM_AW-1:0] dbg_st_addr,
  output [63:0]        dbg_st_data,

  input  [31:0]        imem_rdata,
  output [IMEM_AW-1:0] imem_raddr,
  output               dmem_we,
  output [DMEM_AW-1:0] dmem_waddr,
  output [63:0]        dmem_wdata,
  output [DMEM_AW-1:0] dmem_raddr,
  input  [63:0]        dmem_rdata
);

  reg [IMEM_AW-1:0] pc_fetch;
  reg [IMEM_AW-1:0] pc_exec;

  reg kill;

  reg [9:0] tid;

  reg        ld_wait;
  reg [4:0]  ld_dst;

  reg        pending;
  reg [4:0]  pend_dst;

  wire stall;
  assign stall = ld_wait | pending;

  wire [31:0] ir_raw;
  assign ir_raw = imem_rdata;

  wire [31:0] ir;
  assign ir = kill ? 32'h00000000 : ir_raw;

  wire [3:0] op;
  wire [2:0] rd3;
  wire [2:0] rs1_3;
  wire [2:0] rs2_3;
  wire [1:0] bsel;
  wire [15:0] imm16;

  assign op    = ir[31:28];
  assign rd3   = ir[27:25];
  assign rs1_3 = ir[24:22];
  assign rs2_3 = ir[21:19];
  assign bsel  = ir[18:17];
  assign imm16 = ir[15:0];

  wire signed [15:0] imm16_s;
  assign imm16_s = imm16;

  wire [4:0] rd;
  wire [4:0] rs1;
  wire [4:0] rs2;
  assign rd  = {2'b00, rd3};
  assign rs1 = {2'b00, rs1_3};
  assign rs2 = {2'b00, rs2_3};

  wire [4:0] csrc;
  assign csrc = imm16[4:0];

  wire [DMEM_AW-1:0] base_sel;
  assign base_sel = (bsel == 2'd0) ? a_base :
                    (bsel == 2'd1) ? b_base :
                                    c_base;

  wire [DMEM_AW-1:0] tid_ext;
  assign tid_ext = tid[DMEM_AW-1:0];

  wire [DMEM_AW-1:0] eff_addr_word;
  assign eff_addr_word = base_sel + tid_ext + imm16_s[DMEM_AW-1:0];

  wire [63:0] rs1_data;
  wire [63:0] rs2_data;
  wire [63:0] c_data;

  reg  [4:0]  rf_waddr;
  reg  [63:0] rf_wdata;
  reg         rf_we;

  gpu_regfile_64 rf0 (
    .clk(clk),
    .rst(rst),
    .ra0(rs1),
    .rd0(rs1_data),
    .ra1(rs2),
    .rd1(rs2_data),
    .ra2(csrc),
    .rd2(c_data),
    .we(rf_we),
    .wa(rf_waddr),
    .wd(rf_wdata)
  );

  wire [63:0] i16_add_y;
  wire [63:0] i16_sub_y;
  wire [63:0] i16_relu_y;

  gpu_exec_i16 u_add(.op(2'd0), .a(rs1_data), .b(rs2_data), .y(i16_add_y));
  gpu_exec_i16 u_sub(.op(2'd1), .a(rs1_data), .b(rs2_data), .y(i16_sub_y));
  gpu_exec_i16 u_relu(.op(2'd2), .a(rs1_data), .b(64'd0), .y(i16_relu_y));

  reg  tensor_in_valid;
  reg  [1:0] tensor_in_op;
  reg  [63:0] tensor_a;
  reg  [63:0] tensor_b;
  reg  [63:0] tensor_c;
  wire tensor_out_valid;
  wire [63:0] tensor_out;

  gpu_tensor_unit tensor0(
    .clk(clk),
    .rst(rst),
    .in_valid(tensor_in_valid),
    .in_op(tensor_in_op),
    .in_a(tensor_a),
    .in_b(tensor_b),
    .in_c(tensor_c),
    .out_valid(tensor_out_valid),
    .out_y(tensor_out)
  );

  reg [IMEM_AW-1:0] pc_fetch_next;
  reg take_ctrl;

  always @(*) begin
    rf_we    = 1'b0;
    rf_waddr = rd;
    rf_wdata = 64'd0;

    tensor_in_valid = 1'b0;
    tensor_in_op    = 2'd0;
    tensor_a        = 64'd0;
    tensor_b        = 64'd0;
    tensor_c        = 64'd0;

    pc_fetch_next = pc_fetch + {{(IMEM_AW-1){1'b0}},1'b1};
    take_ctrl     = 1'b0;

    case (op)
      4'h0: begin
      end

      4'h1: begin
        rf_we    = 1'b1;
        rf_wdata = {48'd0, imm16};
      end

      4'h2: begin
        // LOAD is a 2-cycle op with synchronous DMEM read.
        // Hold fetch PC this cycle; advance it in the ld_wait cycle.
        pc_fetch_next = pc_fetch;
      end

      4'h3: begin
      end

      4'h4: begin
        rf_we    = 1'b1;
        rf_wdata = i16_add_y;
      end

      4'h5: begin
        rf_we    = 1'b1;
        rf_wdata = i16_sub_y;
      end

      4'h7: begin
        rf_we    = 1'b1;
        rf_wdata = i16_relu_y;
      end

      4'h8: begin
      end

      4'h9: begin
      end

      4'hA: begin
        if (tid < n_words) begin
          pc_fetch_next = ir[IMEM_AW-1:0];
          take_ctrl     = 1'b1;
        end
      end

      4'hB: begin
        // Tensor op is multi-cycle; hold fetch PC this cycle; advance when result returns.
        pc_fetch_next   = pc_fetch;
        tensor_in_valid = run & (~stall) & (~done);
        tensor_in_op    = 2'd0;
        tensor_a        = rs1_data;
        tensor_b        = rs2_data;
        tensor_c        = 64'd0;
      end

      4'hC: begin
        // Tensor op is multi-cycle; hold fetch PC this cycle; advance when result returns.
        pc_fetch_next   = pc_fetch;
        tensor_in_valid = run & (~stall) & (~done);
        tensor_in_op    = 2'd1;
        tensor_a        = rs1_data;
        tensor_b        = rs2_data;
        tensor_c        = c_data;
      end

      4'hD: begin
      end

      4'hE: begin
        pc_fetch_next = ir[IMEM_AW-1:0];
        take_ctrl     = 1'b1;
      end
      4'hF: begin
      end

      default: begin
      end
    endcase

    if (ld_wait) begin
      rf_we    = 1'b1;
      rf_waddr = ld_dst;
      rf_wdata = dmem_rdata;
    end

    if (pending && tensor_out_valid) begin
      rf_we    = 1'b1;
      rf_waddr = pend_dst;
      rf_wdata = tensor_out;
    end
  end

  assign imem_raddr = pc_fetch;

  assign dmem_raddr = eff_addr_word;
  assign dmem_we    = (run && !stall && !done && (op == 4'h3));
  assign dmem_waddr = eff_addr_word;
  assign dmem_wdata = rs2_data;

  assign dbg_pc       = pc_exec;
  assign dbg_pcf      = pc_fetch;
  assign dbg_ir       = ir;
  assign dbg_tid      = tid;
  assign dbg_ld_wait  = ld_wait;
  assign dbg_pending  = pending;
  assign dbg_eff_addr = eff_addr_word;
  assign dbg_rf_we    = rf_we;
  assign dbg_rf_rd    = rf_waddr;
  assign dbg_rf_wdata = rf_wdata;
  assign dbg_st_we    = dmem_we;
  assign dbg_st_addr  = dmem_waddr;
  assign dbg_st_data  = dmem_wdata;

  always @(posedge clk) begin
    if (rst) begin
      pc_fetch <= {IMEM_AW{1'b0}};
      pc_exec  <= {IMEM_AW{1'b0}};
      kill     <= 1'b0;
      tid      <= 10'd0;
      done     <= 1'b0;
      ld_wait  <= 1'b0;
      ld_dst   <= 5'd0;
      pending  <= 1'b0;
      pend_dst <= 5'd0;
    end else begin
      if (!stall) begin
        pc_exec <= pc_fetch;
        if (kill) kill <= 1'b0;
        if (run && !done && take_ctrl) kill <= 1'b1;
      end

      if (run && !stall && !done) begin
        if (kill) pc_fetch <= pc_fetch;
        else      pc_fetch <= pc_fetch_next;
      end

      if (run && !stall && !done) begin
        if (op == 4'h8) tid <= ir[9:0];
        else if (op == 4'h9) tid <= tid + 10'd1;

        if (op == 4'h2) begin
          ld_wait <= 1'b1;
          ld_dst  <= rd;
        end

        if (op == 4'hB || op == 4'hC) begin
          pending  <= 1'b1;
          pend_dst <= rd;
        end

        if (op == 4'hD) done <= 1'b0;
        if (op == 4'hF) done <= 1'b1;
      end else if (ld_wait) begin
        // load data is written back in this (stalled) cycle; advance fetch PC here.
        ld_wait <= 1'b0;
        if (run && !done) pc_fetch <= pc_fetch + {{(IMEM_AW-1){1'b0}},1'b1};
      end

      if (pending && tensor_out_valid) begin
        // tensor result returns while we are stalled; advance fetch PC here so
        // the next instruction can execute on the following cycle.
        pending <= 1'b0;
        if (run && !done) pc_fetch <= pc_fetch + {{(IMEM_AW-1){1'b0}},1'b1};
      end
    end
  end

endmodule

module gpu_datapath #(
  parameter UDP_REG_SRC_WIDTH = 2,
  parameter IMEM_DEPTH = 256,
  parameter DMEM_DEPTH = 1024,
  parameter IMEM_AW = 8,
  parameter DMEM_AW = 10,
  parameter GPU_ADDR_PREFIX = 8'h7F
)(
  input                               clk,
  input                               reset,

  input                               reg_req_in,
  input                               reg_ack_in,
  input                               reg_rd_wr_L_in,
  input  [`UDP_REG_ADDR_WIDTH-1:0]    reg_addr_in,
  input  [`CPCI_NF2_DATA_WIDTH-1:0]   reg_data_in,
  input  [UDP_REG_SRC_WIDTH-1:0]      reg_src_in,

  output reg                          reg_req_out,
  output reg                          reg_ack_out,
  output reg                          reg_rd_wr_L_out,
  output reg [`UDP_REG_ADDR_WIDTH-1:0]   reg_addr_out,
  output reg [`CPCI_NF2_DATA_WIDTH-1:0]  reg_data_out,
  output reg [UDP_REG_SRC_WIDTH-1:0]     reg_src_out
);

  localparam OFF_CONTROL     = 8'h00;
  localparam OFF_A_BASE      = 8'h01;
  localparam OFF_B_BASE      = 8'h02;
  localparam OFF_C_BASE      = 8'h03;
  localparam OFF_N_WORDS     = 8'h04;
  localparam OFF_IMEM_ADDR   = 8'h10;
  localparam OFF_IMEM_WDATA  = 8'h11;
  localparam OFF_IMEM_WE     = 8'h12;
  localparam OFF_DMEM_ADDR   = 8'h20;
  localparam OFF_DMEM_WLO    = 8'h21;
  localparam OFF_DMEM_WHI    = 8'h22;
  localparam OFF_DMEM_WE     = 8'h23;
  localparam OFF_DMEM_RLO    = 8'h24;
  localparam OFF_DMEM_RHI    = 8'h25;
  localparam OFF_DBG_PC      = 8'h30;
  localparam OFF_DBG_IR      = 8'h31;
  localparam OFF_DBG_TID     = 8'h32;
  localparam OFF_DBG_LASTST  = 8'h33;

  wire [7:0] addr_prefix;
  wire [7:0] addr_off;
  wire addr_hit;

  assign addr_prefix = reg_addr_in[22:15];
  assign addr_off    = reg_addr_in[7:0];
  assign addr_hit = (addr_prefix == GPU_ADDR_PREFIX);

  reg [31:0] a_base;
  reg [31:0] b_base;
  reg [31:0] c_base;
  reg [31:0] n_words;

  reg [IMEM_AW-1:0] imem_addr;
  reg [31:0]        imem_wdata;
  reg               imem_we_pulse;

  reg [DMEM_AW-1:0] dmem_addr;
  reg [31:0]        dmem_wlo;
  reg [31:0]        dmem_whi;
  reg               dmem_we_pulse;

  reg run_reg;

  wire [IMEM_AW-1:0] dbg_pc;
  wire [IMEM_AW-1:0] dbg_pcf;
  wire [31:0]        dbg_ir;
  wire [9:0]         dbg_tid;
  wire               dbg_ld_wait;
  wire               dbg_pending;
  wire [DMEM_AW-1:0] dbg_eff_addr;
  wire               dbg_rf_we;
  wire [4:0]         dbg_rf_rd;
  wire [63:0]        dbg_rf_wdata;
  wire               dbg_st_we;
  wire [DMEM_AW-1:0] dbg_st_addr;
  wire [63:0]        dbg_st_data;

  wire [IMEM_AW-1:0] core_imem_raddr;
  wire [31:0]        core_imem_rdata;

  wire core_dmem_we;
  wire [DMEM_AW-1:0] core_dmem_waddr;
  wire [63:0]        core_dmem_wdata;
  wire [DMEM_AW-1:0] core_dmem_raddr;
  wire [63:0]        core_dmem_rdata;
  wire [63:0]        dmem_dbg_rdata;

  gpu_imem #(.DEPTH(IMEM_DEPTH), .AW(IMEM_AW)) u_imem (
    .clk(clk),
    .raddr(core_imem_raddr),
    .rdata(core_imem_rdata),
    .we(imem_we_pulse),
    .waddr(imem_addr),
    .wdata(imem_wdata)
  );

  gpu_dmem #(.DEPTH(DMEM_DEPTH), .AW(DMEM_AW)) u_dmem (
    .clk(clk),
    .we(dmem_we_pulse | core_dmem_we),
    .waddr(dmem_we_pulse ? dmem_addr : core_dmem_waddr),
    .wdata(dmem_we_pulse ? {dmem_whi, dmem_wlo} : core_dmem_wdata),
    .raddr(core_dmem_raddr),
    .rdata(core_dmem_rdata),
    .dbg_raddr(dmem_addr),
    .dbg_rdata(dmem_dbg_rdata)
  );

  gpu_core #(
    .IMEM_DEPTH(IMEM_DEPTH),
    .DMEM_DEPTH(DMEM_DEPTH),
    .IMEM_AW(IMEM_AW),
    .DMEM_AW(DMEM_AW)
  ) u_core (
    .clk(clk),
    .rst(reset),
    .run(run_reg),
    .n_words(n_words[9:0]),
    .a_base(a_base[DMEM_AW-1:0]),
    .b_base(b_base[DMEM_AW-1:0]),
    .c_base(c_base[DMEM_AW-1:0]),
    .done(),

    .dbg_pc(dbg_pc),
    .dbg_pcf(dbg_pcf),
    .dbg_ir(dbg_ir),
    .dbg_tid(dbg_tid),
    .dbg_ld_wait(dbg_ld_wait),
    .dbg_pending(dbg_pending),
    .dbg_eff_addr(dbg_eff_addr),
    .dbg_rf_we(dbg_rf_we),
    .dbg_rf_rd(dbg_rf_rd),
    .dbg_rf_wdata(dbg_rf_wdata),
    .dbg_st_we(dbg_st_we),
    .dbg_st_addr(dbg_st_addr),
    .dbg_st_data(dbg_st_data),

    .imem_rdata(core_imem_rdata),
    .imem_raddr(core_imem_raddr),
    .dmem_we(core_dmem_we),
    .dmem_waddr(core_dmem_waddr),
    .dmem_wdata(core_dmem_wdata),
    .dmem_raddr(core_dmem_raddr),
    .dmem_rdata(core_dmem_rdata)
  );

  wire handle_req;
  assign handle_req = reg_req_in & (~reg_ack_in) & addr_hit;

  reg [31:0] rd_data;

  always @(*) begin
    rd_data = 32'h0;
    case (addr_off)
      OFF_CONTROL:    rd_data = {31'd0, run_reg};
      OFF_A_BASE:     rd_data = a_base;
      OFF_B_BASE:     rd_data = b_base;
      OFF_C_BASE:     rd_data = c_base;
      OFF_N_WORDS:    rd_data = n_words;
      OFF_IMEM_ADDR:  rd_data = {{(32-IMEM_AW){1'b0}}, imem_addr};
      OFF_IMEM_WDATA: rd_data = imem_wdata;
      OFF_DMEM_ADDR:  rd_data = {{(32-DMEM_AW){1'b0}}, dmem_addr};
      OFF_DMEM_WLO:   rd_data = dmem_wlo;
      OFF_DMEM_WHI:   rd_data = dmem_whi;
      OFF_DMEM_RLO:   rd_data = dmem_dbg_rdata[31:0];
      OFF_DMEM_RHI:   rd_data = dmem_dbg_rdata[63:32];
      OFF_DBG_PC:     rd_data = {{24{1'b0}}, dbg_pc};
      OFF_DBG_IR:     rd_data = dbg_ir;
      OFF_DBG_TID:    rd_data = {{22{1'b0}}, dbg_tid};
      OFF_DBG_LASTST: rd_data = {{(32-DMEM_AW){1'b0}}, dbg_st_addr};
      default:        rd_data = 32'h0;
    endcase
  end

  always @(*) begin
    reg_req_out = reg_req_in;
    reg_ack_out = reg_ack_in;
    reg_rd_wr_L_out = reg_rd_wr_L_in;
    reg_addr_out = reg_addr_in;
    reg_data_out = reg_data_in;
    reg_src_out  = reg_src_in;

    if (handle_req) begin
      reg_ack_out = 1'b1;
      if (reg_rd_wr_L_in) reg_data_out = rd_data;
      else reg_data_out = 32'h0;
    end
  end

  always @(posedge clk) begin
    if (reset) begin
      a_base <= 32'd0;
      b_base <= 32'd0;
      c_base <= 32'd0;
      n_words <= 32'd0;
      imem_addr <= {IMEM_AW{1'b0}};
      imem_wdata <= 32'd0;
      imem_we_pulse <= 1'b0;
      dmem_addr <= {DMEM_AW{1'b0}};
      dmem_wlo <= 32'd0;
      dmem_whi <= 32'd0;
      dmem_we_pulse <= 1'b0;
      run_reg <= 1'b0;
    end else begin
      imem_we_pulse <= 1'b0;
      dmem_we_pulse <= 1'b0;

      if (handle_req && (reg_rd_wr_L_in == 1'b0)) begin
        case (addr_off)
          OFF_CONTROL:    run_reg <= reg_data_in[0];
          OFF_A_BASE:     a_base <= reg_data_in;
          OFF_B_BASE:     b_base <= reg_data_in;
          OFF_C_BASE:     c_base <= reg_data_in;
          OFF_N_WORDS:    n_words <= reg_data_in;
          OFF_IMEM_ADDR:  imem_addr <= reg_data_in[IMEM_AW-1:0];
          OFF_IMEM_WDATA: imem_wdata <= reg_data_in;
          OFF_IMEM_WE:    imem_we_pulse <= reg_data_in[0];
          OFF_DMEM_ADDR:  dmem_addr <= reg_data_in[DMEM_AW-1:0];
          OFF_DMEM_WLO:   dmem_wlo <= reg_data_in;
          OFF_DMEM_WHI:   dmem_whi <= reg_data_in;
          OFF_DMEM_WE:    dmem_we_pulse <= reg_data_in[0];
          default: begin end
        endcase
      end
    end
  end

endmodule
