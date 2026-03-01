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